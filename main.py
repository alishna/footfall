from fastapi import FastAPI, HTTPException, Depends, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO


# g_username = None
# g_email = None
# g_password = None
# g_confirm_password = None

app = FastAPI()
class User(BaseModel):
    username: str
    password: str
    

app.mount("/html", StaticFiles(directory="html/"), name="html")


@app.get("/")
async def root():
    try:
        with open("html/index.html", "r") as file:
            html_content = file.read()
            return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="found")

@app.post("/register")
async def register_user(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...)):
    
    global g_username
    global g_email
    global g_password
    global g_confirm_password
    
    g_username = username
    g_email = email
    g_password = password
    g_confirm_password = confirm_password
    
    try:
        with open("html/login.html", "r") as file:
            html_content = file.read()
            return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="found")

@app.post("/login")
async def login_user(
    username: str = Form(...),
    password: str = Form(...), ):
    
    print(username, g_username)
    
    if username == g_username and password == g_password:
        try:
            with open("html/overall.html", "r") as file:
                html_content = file.read()
                return HTMLResponse(content=html_content, status_code=200)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="found")
    
    else:
        return {"error": "Invalid username or password"}

    

@app.get("/authenticate")
async def authenticate(details: User):
    if (g_username == details.username and g_password == details.password):
        return {"message": "Authentication successful"}
    else:
        return {"message": "Authentication failed"}
    


# Tracker route
@app.get("/tracker")
async def tracker():
    try:
        with open("html/tracking.html", "r") as file:
            html_content = file.read()
            return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Tracker.html not found")