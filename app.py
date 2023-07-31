from fastapi import FastAPI
from api.disease import disease_router
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from starlette_context import plugins


def create_app() -> FastAPI:
    """Creates an instance of FastApi App"""
    app = FastAPI(
        title="Fettle",
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            ),
        ],
    )
    app.include_router(disease_router, tags=["disease"])
    return app


app = create_app()
