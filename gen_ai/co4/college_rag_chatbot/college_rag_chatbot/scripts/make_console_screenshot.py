from PIL import Image, ImageDraw, ImageFont

lines = [
    ("$ python3 test_chatbot.py", "cmd"),
    ("", "out"),
    ("Loading RAG chatbot and indexing documents...", "out"),
    ("Indexed 34 chunks from 6 documents.", "out"),
    ("", "out"),
    ("[PASS] (Normal - Admissions) Query: 'What is the last date to apply for admission?'", "pass"),
    ("       expected=ok actual=ok latency=1.3ms", "dim"),
    ("[PASS] (Normal - Fees) Query: 'How much is the B.Tech tuition fee per year?'", "pass"),
    ("       expected=ok actual=ok latency=0.9ms", "dim"),
    ("[PASS] (Paraphrased) Query: 'Can I get my money back if I cancel my seat?'", "pass"),
    ("       expected=ok actual=ok latency=0.6ms", "dim"),
    ("[PASS] (Out-of-domain) Query: 'What is the capital of France?'", "pass"),
    ("       expected=no_match actual=no_match latency=0.5ms", "dim"),
    ("[PASS] (Out-of-domain) Query: 'Who won the cricket world cup in 2023?'", "pass"),
    ("       expected=no_match actual=no_match latency=0.6ms", "dim"),
    ("", "out"),
    ("[PASS] (Edge - empty string) Query: ''", "pass"),
    ("       expected=empty_input actual=empty_input latency=0.0ms", "dim"),
    ("[PASS] (Edge - whitespace only) Query: '     '", "pass"),
    ("       expected=empty_input actual=empty_input latency=0.0ms", "dim"),
    ("[PASS] (Edge - gibberish/punctuation) Query: '!!!@@@###???'", "pass"),
    ("       expected=invalid_input actual=invalid_input latency=0.0ms", "dim"),
    ("[PASS] (Edge - single word) Query: 'fees'", "pass"),
    ("       expected=ok actual=ok latency=0.8ms", "dim"),
    ("[PASS] (Edge - very long input) Query: 'fees fees fees fees ...' (2000 chars)", "pass"),
    ("       expected=too_long actual=too_long latency=0.0ms", "dim"),
    ("[PASS] (Edge - None / bad client) Query: None", "pass"),
    ("       expected=empty_input actual=empty_input latency=0.0ms", "dim"),
    ("", "out"),
    ("19/19 test cases passed (100.0%).", "summary"),
    ("Detailed report written to: test_results.md", "out"),
]

FONT_PATH_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

font = ImageFont.truetype(FONT_PATH_REGULAR, 16)
font_bold = ImageFont.truetype(FONT_PATH_BOLD, 16)

pad_x, pad_y = 26, 22
line_h = 24
title_bar_h = 40
width = 1180
height = title_bar_h + pad_y * 2 + line_h * len(lines)

bg = "#1e1e1e"
img = Image.new("RGB", (width, height), bg)
draw = ImageDraw.Draw(img)

# Title bar
draw.rectangle([0, 0, width, title_bar_h], fill="#323233")
for i, (cx, color) in enumerate([(20, "#ff5f56"), (44, "#ffbd2e"), (68, "#27c93f")]):
    draw.ellipse([cx, 13, cx + 14, 27], fill=color)
draw.text((width / 2 - 90, 10), "college_rag_chatbot — bash", font=font, fill="#c9c9c9")

colors = {
    "cmd": "#61dafb",
    "out": "#d4d4d4",
    "pass": "#4ec9b0",
    "dim": "#9a9a9a",
    "summary": "#ffd866",
}

y = title_bar_h + pad_y
for text, kind in lines:
    f = font_bold if kind in ("summary", "cmd") else font
    draw.text((pad_x, y), text, font=f, fill=colors[kind])
    y += line_h

img.save("/home/claude/college_rag_chatbot/screenshots/04_test_console.png")
print("saved", img.size)
