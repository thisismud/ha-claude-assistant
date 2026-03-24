  from fastapi import FastAPI
  import os

  app = FastAPI()

  @app.get("/")
  def root():
      return {"status": "AI assistant running"}
