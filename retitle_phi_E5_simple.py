from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


output = Path("phi_anisotropy_out/phi_E5_simple.png")
source = Path("phi_anisotropy_out/phi_E5_simple_original.png")

if not source.exists():
    source.write_bytes(output.read_bytes())

plot = Image.open(source).convert("RGB")
top_margin = 55
image = Image.new("RGB", (plot.width, plot.height + top_margin), "white")
image.paste(plot, (0, top_margin))

font_path = font_manager.findfont("DejaVu Sans")
title_font = ImageFont.truetype(font_path, 24)
label_font = ImageFont.truetype(font_path, 21)
draw = ImageDraw.Draw(image)

title = "High-energy muon momentum φ (E > 5 GeV)"
title_box = draw.textbbox((0, 0), title, font=title_font)
title_width = title_box[2] - title_box[0]
draw.text(((image.width - title_width) / 2, 12), title, fill="black", font=title_font)

draw.rectangle((0, top_margin + 130, 52, top_margin + 460), fill="white")

label = "Muon count"
label_box = label_font.getbbox(label)
label_image = Image.new(
    "RGBA",
    (label_box[2] - label_box[0] + 8, label_box[3] - label_box[1] + 8),
    (255, 255, 255, 0),
)
ImageDraw.Draw(label_image).text((4, 4 - label_box[1]), label, fill="black", font=label_font)
label_image = label_image.rotate(90, expand=True)
label_y = top_margin + 294 - label_image.height // 2
image.paste(label_image, (18, label_y), label_image)

image.save(output, dpi=(150, 150))
