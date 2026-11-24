import os, re
import gradio as gr

p = os.path.join(os.path.dirname(gr.__file__), 'templates', 'frontend', 'assets', 'Index-DRTsT41f.js')
s = open(p, encoding='utf-8').read()

idx = s.find('label:"Brush Size"')
print(s[idx-200:idx+200])
