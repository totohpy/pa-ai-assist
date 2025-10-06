import streamlit as st
from datetime import datetime
import re
from openai import OpenAI
import os
import html
import io
import docx
from docx.enum.section import WD_ORIENT
from docx.shared import Pt
import base64
import json
import streamlit.components.v1 as components

# (วางโค้ดทั้งหมดจากไฟล์ 2_🤖_Plan_Generator.py ที่ใช้งานได้ดีล่าสุดของคุณที่นี่)
# ... โค้ดที่สมบูรณ์ที่สุดจากคำตอบก่อนหน้า ...
