from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile
from fastapi.staticfiles import StaticFiles
from app.routers import chat, route_planning, image_recognition, video_tracking, image_analysis_chat
import uvicorn
from typing import Dict
import platform
import sys
import datetime
import logging
from contextlib import asynccontextmanager
import signal
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    """处理进程信号"""
    logger.info(f"收到信号: {signum}")
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # 终止信号

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    处理应用程序的启动和关闭事件
    """
    logger.info("应用程序启动...")
    try:
        # 启动时的初始化代码
        yield
    except Exception as e:
        logger.error(f"应用程序启动错误: {str(e)}")
        raise
    finally:
        # 关闭时的清理代码
        logger.info("正在清理资源...")
        try:
            # 在这里添加任何需要的清理代码
            # 例如关闭数据库连接、释放模型资源等
            pass
        except Exception as e:
            logger.error(f"清理资源时出错: {str(e)}")
        logger.info("应用程序关闭完成")

app = FastAPI(
    title="AI Assistant API",
    description="智能路径规划与目标追踪系统API",
    version="1.0.0",
    docs_url="/api/docs",  # 修改Swagger文档路径
    redoc_url="/api/redoc",  # 修改ReDoc文档路径
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3001"],  # 添加你的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# 挂载静态文件目录
app.mount("/output", StaticFiles(directory="Main/output"), name="output")

# 注册路由
app.include_router(chat.router, prefix="/api/chat", tags=["聊天"])
app.include_router(route_planning.router, prefix="/api/route", tags=["路径规划"])
app.include_router(image_recognition.router, prefix="/api/image-recognition", tags=["图像识别"])
app.include_router(video_tracking.router, prefix="/api/video-tracking", tags=["视频追踪"])
app.include_router(image_analysis_chat.router, prefix="/api/image-analysis-chat", tags=["图片分析聊天"])

@app.get("/", tags=["系统信息"])
async def root() -> Dict:
    """获取系统基本信息"""
    try:
        return {
            "message": "Welcome to AI Assistant API",
            "version": "1.0.0",
            "system_info": {
                "python_version": sys.version,
                "platform": platform.platform()
            },
            "api_docs": {
                "swagger": "/api/docs",
                "redoc": "/api/redoc"
            }
        }
    except Exception as e:
        logger.error(f"获取系统信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取系统信息失败")

@app.get("/health", tags=["健康检查"])
async def health_check() -> Dict:
    """系统健康检查"""
    try:
        # 这里可以添加数据库连接检查等
        return {
            "status": "healthy",
            "services": {
                "api": "up",
                "database": "up",
                "ollama": "up"  # 本地大模型服务状态
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/test", tags=["测试"])
async def test():
    """测试接口"""
    try:
        return {"status": "ok", "message": "API服务正常运行"}
    except Exception as e:
        logger.error(f"测试接口失败: {str(e)}")
        raise HTTPException(status_code=500, detail="测试接口失败")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"全局异常: {str(exc)}")
    return {
        "success": False,
        "error": "服务器内部错误",
        "detail": str(exc),
        "timestamp": datetime.datetime.now().isoformat()
    }

if __name__ == "__main__":
    logger.info("启动服务器...")
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            workers=1,
            timeout_keep_alive=400,  # 增加保持连接的超时时间
            limit_concurrency=10,    # 限制并发连接数
            timeout=400              # 增加请求超时时间
        )
    except KeyboardInterrupt:
        logger.info("正在优雅地关闭服务器...")
    except Exception as e:
        logger.error(f"服务器异常关闭: {str(e)}")
    finally:
        logger.info("服务器已关闭")

# ... 其他路由和代码 