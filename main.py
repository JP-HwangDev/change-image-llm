import os
import uuid
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline

app = FastAPI()

# 템플릿 디렉토리 및 정적 파일 설정
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 임시 파일 저장 폴더 생성
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


# 애플리케이션 시작 시 모델 로드
@app.on_event("startup")
async def load_model():
    global pipe
    model_id = "nitrosocke/Ghibli-Diffusion"  # 또는 다른 지브리 LoRA 모델 사용 가능
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 로드 실패: {e}")
    pipe.to(device)
    pipe.enable_attention_slicing()  # 메모리 최적화


# 홈 페이지: 업로드 폼을 렌더링
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 변환 엔드포인트: 파일 업로드 후 이미지 변환 실행, 결과를 temp 폴더에 저장하고 uid 반환
@app.post("/convert/")
async def convert_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        original_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="이미지 읽기 실패")

    # 모델 요구에 맞게 이미지 크기 조정 (512x512)
    original_image = original_image.resize((512, 512))

    # 고유 uid 생성 및 파일 경로 설정
    uid = str(uuid.uuid4())
    original_path = os.path.join(TEMP_DIR, f"{uid}_original.png")
    processed_path = os.path.join(TEMP_DIR, f"{uid}_processed.png")

    # 원본 이미지 저장
    original_image.save(original_path, format="PNG")

    # 지브리 스타일 변환: 원본 구조를 유지하며 스타일만 변경하도록 프롬프트 사용
    prompt: str = "A natural, hand-drawn cartoon illustration in Studio Ghibli style, featuring soft pastel colors, smooth and delicate brush strokes, a gentle and whimsical atmosphere, and a magical natural setting"
    strength = 0.5
    guidance_scale = 12.0
    num_inference_steps = 100

    try:
        processed_image = pipe(
            prompt=prompt,
            image=original_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 변환 실패: {e}")

    # 변환된 이미지 저장
    processed_image.save(processed_path, format="PNG")

    # uid 반환 (프론트엔드에서 다운로드 링크 생성 시 사용)
    return JSONResponse({"uid": uid})


# 다운로드 엔드포인트: processed, original, compare(비교) 옵션 지원
@app.get("/download/{uid}/{kind}")
async def download_image(uid: str, kind: str):
    if kind == "processed":
        path = os.path.join(TEMP_DIR, f"{uid}_processed.png")
    elif kind == "original":
        path = os.path.join(TEMP_DIR, f"{uid}_original.png")
    elif kind == "compare":
        # 원본과 변환 이미지를 가로로 결합
        orig_path = os.path.join(TEMP_DIR, f"{uid}_original.png")
        proc_path = os.path.join(TEMP_DIR, f"{uid}_processed.png")
        if not os.path.exists(orig_path) or not os.path.exists(proc_path):
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
        original_image = Image.open(orig_path)
        processed_image = Image.open(proc_path)
        width = original_image.width + processed_image.width
        height = max(original_image.height, processed_image.height)
        combined = Image.new("RGB", (width, height))
        combined.paste(original_image, (0, 0))
        combined.paste(processed_image, (original_image.width, 0))
        buf = io.BytesIO()
        combined.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png",
                                 headers={"Content-Disposition": f"attachment; filename={uid}_compare.png"})
    else:
        raise HTTPException(status_code=400, detail="잘못된 요청")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

    return FileResponse(path, media_type="image/png", filename=os.path.basename(path))
