"""Generate a 4-slide presentation for the IAP Pipeline codebase."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── Color palette (light theme) ──
BG_LIGHT   = RGBColor(0xF5, 0xF6, 0xFA)
BG_CARD    = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT     = RGBColor(0x00, 0x78, 0xD4)
ACCENT2    = RGBColor(0x6B, 0x3F, 0xA0)
TEXT_DARK  = RGBColor(0x1E, 0x1E, 0x2E)
TEXT_MED   = RGBColor(0x55, 0x55, 0x66)
GREEN      = RGBColor(0x0E, 0x8A, 0x4E)
ORANGE     = RGBColor(0xD8, 0x7B, 0x00)

def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_shape(slide, left, top, width, height, fill_color, border_color=None, radius=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)
    else:
        shape.line.fill.background()
    return shape

def add_text(slide, left, top, width, height, text, font_size=18, color=TEXT_DARK, bold=False, alignment=PP_ALIGN.LEFT, font_name="Segoe UI"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox

def add_bullet_text(slide, left, top, width, height, items, font_size=16, color=TEXT_MED, bullet_color=ACCENT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(6)
        p.space_after = Pt(4)
        # Bullet character
        run_bullet = p.add_run()
        run_bullet.text = "▸ "
        run_bullet.font.size = Pt(font_size)
        run_bullet.font.color.rgb = bullet_color
        run_bullet.font.name = "Segoe UI"
        # Item text
        run_text = p.add_run()
        run_text.text = item
        run_text.font.size = Pt(font_size)
        run_text.font.color.rgb = color
        run_text.font.name = "Segoe UI"
    return txBox

# ════════════════════════════════════════════════════════════════════
# SLIDE 1 – Title Slide
# ════════════════════════════════════════════════════════════════════
slide1 = prs.slides.add_slide(prs.slide_layouts[6])  # blank
set_slide_bg(slide1, BG_LIGHT)

# Accent bar at top
add_shape(slide1, Inches(0), Inches(0), Inches(13.333), Inches(0.08), ACCENT)

# Title
add_text(slide1, Inches(1.2), Inches(1.8), Inches(11), Inches(1.2),
         "Intent-Aware Prompting (IAP) Pipeline",
         font_size=44, color=TEXT_DARK, bold=True, alignment=PP_ALIGN.CENTER)

# Subtitle
add_text(slide1, Inches(1.5), Inches(3.1), Inches(10.3), Inches(0.9),
         "Detecting Psychological Manipulation in Dialogues via LLM-Driven Intent Analysis",
         font_size=22, color=ACCENT, bold=False, alignment=PP_ALIGN.CENTER)

# Divider line
add_shape(slide1, Inches(4.5), Inches(4.2), Inches(4.3), Inches(0.04), ACCENT2)

# Key idea box
add_shape(slide1, Inches(2.5), Inches(4.8), Inches(8.3), Inches(1.4), BG_CARD, ACCENT2)
add_text(slide1, Inches(2.8), Inches(4.95), Inches(7.7), Inches(1.1),
         "Core Idea: Separate \"what is said\" (surface dialogue) from \"why it is said\"\n"
         "(latent intent) to improve manipulation detection with reduced false negatives.",
         font_size=17, color=TEXT_MED, alignment=PP_ALIGN.CENTER)

# Footer
add_text(slide1, Inches(1), Inches(6.6), Inches(11.3), Inches(0.5),
         "Built with Python  ·  Async Pipeline  ·  OpenAI / Anthropic / Groq  ·  MentalManip Dataset",
         font_size=13, color=TEXT_MED, alignment=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════
# SLIDE 2 – Architecture & Pipeline Flow
# ════════════════════════════════════════════════════════════════════
slide2 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide2, BG_LIGHT)

add_text(slide2, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7),
         "Architecture & Pipeline Flow",
         font_size=34, color=TEXT_DARK, bold=True)
add_shape(slide2, Inches(0.6), Inches(0.95), Inches(3), Inches(0.05), ACCENT)

# Step boxes
step_data = [
    ("STEP 1", "Data Loading", "data_loader.py", [
        "Parse MentalManip CSV dataset",
        "Extract dialogue text & ground-truth labels",
        "Handle technique & vulnerability annotations",
    ], Inches(0.4)),
    ("STEP 2", "Intent Extraction", "intent_extractor.py", [
        "Parallel extraction for Person A & B",
        "LLM acts as psycholinguist (Theory of Mind)",
        "Identifies guilt-tripping, gaslighting,\nblameshifting, emotional blackmail, etc.",
    ], Inches(4.6)),
    ("STEP 3", "Classification", "classifier.py", [
        "Combines dialogue + both intent vectors",
        "LLM outputs JSON: manipulative (0/1),\nconfidence, technique, explanation",
        "Results saved to predictions.csv",
    ], Inches(8.8)),
]

for step_label, title, filename, bullets, left in step_data:
    # Card background
    add_shape(slide2, left, Inches(1.5), Inches(3.9), Inches(4.8), BG_CARD, ACCENT2)
    # Step label
    add_shape(slide2, left + Inches(0.2), Inches(1.7), Inches(1.4), Inches(0.45), ACCENT2)
    add_text(slide2, left + Inches(0.25), Inches(1.72), Inches(1.3), Inches(0.4),
             step_label, font_size=13, color=RGBColor(0xFF, 0xFF, 0xFF), bold=True, alignment=PP_ALIGN.CENTER)
    # Title
    add_text(slide2, left + Inches(0.3), Inches(2.3), Inches(3.3), Inches(0.5),
             title, font_size=22, color=ACCENT, bold=True)
    # Filename
    add_text(slide2, left + Inches(0.3), Inches(2.8), Inches(3.3), Inches(0.4),
             filename, font_size=13, color=ORANGE, bold=False)
    # Bullets
    add_bullet_text(slide2, left + Inches(0.3), Inches(3.3), Inches(3.4), Inches(2.8),
                    bullets, font_size=14, color=TEXT_MED)

# Arrows between steps
for arrow_left in [Inches(4.35), Inches(8.55)]:
    add_text(slide2, arrow_left, Inches(3.5), Inches(0.4), Inches(0.5),
             "→", font_size=36, color=ACCENT, bold=True, alignment=PP_ALIGN.CENTER)

# Bottom bar – supporting modules
add_shape(slide2, Inches(0.4), Inches(6.55), Inches(12.5), Inches(0.7), BG_CARD, RGBColor(0xDD, 0xDD, 0xEE))
add_text(slide2, Inches(0.6), Inches(6.6), Inches(12), Inches(0.55),
         "Supporting Modules:   llm_client.py (unified API layer)  ·  prompts.py (prompt templates)  ·  config.py (settings)  ·  evaluation.py (metrics)  ·  pipeline.py (async orchestrator)",
         font_size=13, color=TEXT_MED, alignment=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════
# SLIDE 3 – Key Design Decisions & Technical Details
# ════════════════════════════════════════════════════════════════════
slide3 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide3, BG_LIGHT)

add_text(slide3, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7),
         "Key Design Decisions & Technical Details",
         font_size=34, color=TEXT_DARK, bold=True)
add_shape(slide3, Inches(0.6), Inches(0.95), Inches(3), Inches(0.05), ACCENT)

# Card 1 – Multi-Provider LLM
add_shape(slide3, Inches(0.4), Inches(1.4), Inches(6.2), Inches(2.5), BG_CARD, ACCENT2)
add_text(slide3, Inches(0.7), Inches(1.55), Inches(5.5), Inches(0.5),
         "🔌  Multi-Provider LLM Abstraction", font_size=20, color=ACCENT, bold=True)
add_bullet_text(slide3, Inches(0.7), Inches(2.1), Inches(5.7), Inches(1.7), [
    "Unified call_llm() dispatcher for OpenAI, Anthropic & Groq",
    "Groq uses OpenAI SDK with custom base_url",
    "Exponential backoff retry for rate limits (2→3→5→9s)",
    "Configurable temperature & max tokens per step",
], font_size=14, color=TEXT_MED)

# Card 2 – Prompt Engineering
add_shape(slide3, Inches(6.8), Inches(1.4), Inches(6.2), Inches(2.5), BG_CARD, ACCENT2)
add_text(slide3, Inches(7.1), Inches(1.55), Inches(5.5), Inches(0.5),
         "🧠  Prompt Engineering Strategy", font_size=20, color=ACCENT, bold=True)
add_bullet_text(slide3, Inches(7.1), Inches(2.1), Inches(5.7), Inches(1.7), [
    "System prompt: LLM = expert psycholinguist role",
    "Intent extraction focuses on latent manipulation tactics",
    "Classification combines dialogue + dual intent vectors",
    "Structured JSON output: label, confidence, technique",
], font_size=14, color=TEXT_MED)

# Card 3 – Async Concurrency
add_shape(slide3, Inches(0.4), Inches(4.2), Inches(6.2), Inches(2.5), BG_CARD, ACCENT2)
add_text(slide3, Inches(0.7), Inches(4.35), Inches(5.5), Inches(0.5),
         "⚡  Async Concurrency Model", font_size=20, color=ACCENT, bold=True)
add_bullet_text(slide3, Inches(0.7), Inches(4.9), Inches(5.7), Inches(1.7), [
    "asyncio-based pipeline with configurable semaphore",
    "Parallel intent extraction for Person A & B",
    "Concurrent dialogue processing (default: 5 workers)",
    "Non-blocking API calls maximize throughput",
], font_size=14, color=TEXT_MED)

# Card 4 – Evaluation Focus
add_shape(slide3, Inches(6.8), Inches(4.2), Inches(6.2), Inches(2.5), BG_CARD, ACCENT2)
add_text(slide3, Inches(7.1), Inches(4.35), Inches(5.5), Inches(0.5),
         "📊  Evaluation & Safety Focus", font_size=20, color=ACCENT, bold=True)
add_bullet_text(slide3, Inches(7.1), Inches(4.9), Inches(5.7), Inches(1.7), [
    "Key metric: False Negative Rate (FNR) reduction",
    "Missed manipulation = safety risk → prioritize recall",
    "Full confusion matrix + precision, recall, F1, accuracy",
    "Per-dialogue predictions with explanations saved",
], font_size=14, color=TEXT_MED)

# ════════════════════════════════════════════════════════════════════
# SLIDE 4 – Results & Impact
# ════════════════════════════════════════════════════════════════════
slide4 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide4, BG_LIGHT)

add_text(slide4, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7),
         "Results & Impact",
         font_size=34, color=TEXT_DARK, bold=True)
add_shape(slide4, Inches(0.6), Inches(0.95), Inches(3), Inches(0.05), ACCENT)

# Metric cards row
metrics = [
    ("75.0%", "Accuracy", ACCENT),
    ("88.2%", "Precision", ACCENT2),
    ("83.3%", "Recall", GREEN),
    ("85.7%", "F1 Score", GREEN),
    ("16.7%", "False Neg Rate", ORANGE),
]

for i, (value, label, clr) in enumerate(metrics):
    left = Inches(0.4 + i * 2.6)
    add_shape(slide4, left, Inches(1.4), Inches(2.35), Inches(2.0), BG_CARD, clr)
    add_text(slide4, left, Inches(1.65), Inches(2.35), Inches(0.8),
             value, font_size=38, color=clr, bold=True, alignment=PP_ALIGN.CENTER)
    add_text(slide4, left, Inches(2.45), Inches(2.35), Inches(0.5),
             label, font_size=16, color=TEXT_MED, bold=False, alignment=PP_ALIGN.CENTER)

# Key Takeaways
add_shape(slide4, Inches(0.4), Inches(3.8), Inches(12.5), Inches(3.1), BG_CARD, ACCENT2)
add_text(slide4, Inches(0.7), Inches(3.95), Inches(5), Inches(0.5),
         "Key Takeaways", font_size=22, color=ACCENT, bold=True)

add_bullet_text(slide4, Inches(0.7), Inches(4.5), Inches(5.8), Inches(2.2), [
    "IAP achieves ~30.5% FNR improvement over baselines",
    "Intent separation catches subtle manipulation\nthat surface-level analysis misses",
    "Detected techniques: gaslighting, guilt-tripping,\nemotional blackmail, minimization",
    "High confidence predictions with explanations\nenable human-in-the-loop review",
], font_size=14, color=TEXT_MED)

add_text(slide4, Inches(7), Inches(4.5), Inches(5.5), Inches(0.5),
         "Sample Output (per dialogue):", font_size=16, color=ACCENT, bold=True)

# Sample JSON-like output card
add_shape(slide4, Inches(7), Inches(5.0), Inches(5.5), Inches(1.6), RGBColor(0xF0, 0xF0, 0xF5))
add_text(slide4, Inches(7.2), Inches(5.1), Inches(5.1), Inches(1.4),
         '{\n'
         '  "manipulative": 1,\n'
         '  "confidence": 0.9,\n'
         '  "technique": "Gaslighting, Guilt-tripping",\n'
         '  "explanation": "Person A denies B\'s ..."\n'
         '}',
         font_size=13, color=GREEN, font_name="Consolas")

# ════════════════════════════════════════════════════════════════════
# SLIDE 5 – The Core Paradigm: Memory Over Learning
# ════════════════════════════════════════════════════════════════════
slide5 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide5, BG_LIGHT)

add_text(slide5, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7),
         "The Core Paradigm: Memory Over Learning",
         font_size=34, color=TEXT_DARK, bold=True)
add_shape(slide5, Inches(0.6), Inches(0.95), Inches(3), Inches(0.05), ACCENT)

# Card 1 – The Shift
add_shape(slide5, Inches(0.4), Inches(1.5), Inches(3.9), Inches(4.8), BG_CARD, ACCENT2)
add_shape(slide5, Inches(0.6), Inches(1.7), Inches(1.8), Inches(0.45), ACCENT2)
add_text(slide5, Inches(0.65), Inches(1.72), Inches(1.7), Inches(0.4),
         "THE SHIFT", font_size=13, color=RGBColor(0xFF, 0xFF, 0xFF), bold=True, alignment=PP_ALIGN.CENTER)
add_text(slide5, Inches(0.7), Inches(2.4), Inches(3.3), Inches(0.5),
         "🔄  From Learning to Retrieval", font_size=20, color=ACCENT, bold=True)
add_bullet_text(slide5, Inches(0.7), Inches(3.0), Inches(3.4), Inches(3.0), [
    "Moves away from the traditional ML paradigm of fine-tuning weights",
    "No classification boundary is learned from data",
    "Replaces gradient-based optimization with direct memory lookup",
], font_size=14, color=TEXT_MED)

# Card 2 – The Memory Bank
add_shape(slide5, Inches(4.6), Inches(1.5), Inches(3.9), Inches(4.8), BG_CARD, ACCENT2)
add_shape(slide5, Inches(4.8), Inches(1.7), Inches(2.4), Inches(0.45), ACCENT2)
add_text(slide5, Inches(4.85), Inches(1.72), Inches(2.3), Inches(0.4),
         "THE MEMORY BANK", font_size=13, color=RGBColor(0xFF, 0xFF, 0xFF), bold=True, alignment=PP_ALIGN.CENTER)
add_text(slide5, Inches(4.9), Inches(2.4), Inches(3.3), Inches(0.5),
         "🗄️  Stored Exemplar Database", font_size=20, color=ACCENT, bold=True)
add_bullet_text(slide5, Inches(4.9), Inches(3.0), Inches(3.4), Inches(3.0), [
    "Stores exact, labeled exemplars of both normal and manipulative dialogues",
    "Database serves as the model's entire knowledge base",
    "Each entry preserves full context and ground-truth label",
], font_size=14, color=TEXT_MED)

# Card 3 – The Signal
add_shape(slide5, Inches(8.8), Inches(1.5), Inches(3.9), Inches(4.8), BG_CARD, ACCENT2)
add_shape(slide5, Inches(9.0), Inches(1.7), Inches(1.8), Inches(0.45), ACCENT2)
add_text(slide5, Inches(9.05), Inches(1.72), Inches(1.7), Inches(0.4),
         "THE SIGNAL", font_size=13, color=RGBColor(0xFF, 0xFF, 0xFF), bold=True, alignment=PP_ALIGN.CENTER)
add_text(slide5, Inches(9.1), Inches(2.4), Inches(3.3), Inches(0.5),
         "📐  Semantic Similarity", font_size=20, color=ACCENT, bold=True)
add_bullet_text(slide5, Inches(9.1), Inches(3.0), Inches(3.4), Inches(3.0), [
    "Determines intent purely by measuring semantic similarity",
    "No complex neural layers or probability computations",
    "Distance in vector space replaces learned decision boundaries",
], font_size=14, color=TEXT_MED)

# ════════════════════════════════════════════════════════════════════
# SLIDE 6 – The Retrieval Mechanism
# ════════════════════════════════════════════════════════════════════
slide6 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide6, BG_LIGHT)

add_text(slide6, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7),
         "The Retrieval Mechanism",
         font_size=34, color=TEXT_DARK, bold=True)
add_shape(slide6, Inches(0.6), Inches(0.95), Inches(3), Inches(0.05), ACCENT)

# Card 1 – Vectorization
add_shape(slide6, Inches(0.4), Inches(1.5), Inches(3.9), Inches(4.8), BG_CARD, ACCENT2)
add_shape(slide6, Inches(0.6), Inches(1.7), Inches(2.2), Inches(0.45), ACCENT2)
add_text(slide6, Inches(0.65), Inches(1.72), Inches(2.1), Inches(0.4),
         "VECTORIZATION", font_size=13, color=RGBColor(0xFF, 0xFF, 0xFF), bold=True, alignment=PP_ALIGN.CENTER)
add_text(slide6, Inches(0.7), Inches(2.4), Inches(3.3), Inches(0.5),
         "🔢  Dense Vector Encoding", font_size=20, color=ACCENT, bold=True)
add_bullet_text(slide6, Inches(0.7), Inches(3.0), Inches(3.4), Inches(3.0), [
    "Converts new conversation segments into dense vector representations",
    "Uses embedding models such as sentence-transformers",
    "Captures semantic meaning in high-dimensional space",
], font_size=14, color=TEXT_MED)

# Card 2 – Distance Calculation
add_shape(slide6, Inches(4.6), Inches(1.5), Inches(3.9), Inches(4.8), BG_CARD, ACCENT2)
add_shape(slide6, Inches(4.8), Inches(1.7), Inches(2.8), Inches(0.45), ACCENT2)
add_text(slide6, Inches(4.85), Inches(1.72), Inches(2.7), Inches(0.4),
         "DISTANCE CALCULATION", font_size=13, color=RGBColor(0xFF, 0xFF, 0xFF), bold=True, alignment=PP_ALIGN.CENTER)
add_text(slide6, Inches(4.9), Inches(2.4), Inches(3.3), Inches(0.5),
         "📏  k-NN Search", font_size=20, color=ACCENT, bold=True)
add_bullet_text(slide6, Inches(4.9), Inches(3.0), Inches(3.4), Inches(3.0), [
    "Executes fast k-nearest-neighbor search in vector space",
    "Locates the closest historical examples to the input",
    "Optimized algorithms enable sub-millisecond lookups",
], font_size=14, color=TEXT_MED)

# Card 3 – Classification by Proximity
add_shape(slide6, Inches(8.8), Inches(1.5), Inches(3.9), Inches(4.8), BG_CARD, ACCENT2)
add_shape(slide6, Inches(9.0), Inches(1.7), Inches(3.4), Inches(0.45), ACCENT2)
add_text(slide6, Inches(9.05), Inches(1.72), Inches(3.3), Inches(0.4),
         "CLASSIFICATION BY PROXIMITY", font_size=13, color=RGBColor(0xFF, 0xFF, 0xFF), bold=True, alignment=PP_ALIGN.CENTER)
add_text(slide6, Inches(9.1), Inches(2.4), Inches(3.3), Inches(0.5),
         "🎯  Cluster Matching", font_size=20, color=ACCENT, bold=True)
add_bullet_text(slide6, Inches(9.1), Inches(3.0), Inches(3.4), Inches(3.0), [
    "Flags segment as manipulation if it clusters near stored tactics",
    "Matches prototypical patterns like \"accusation\" or \"feigning innocence\"",
    "Semantic proximity drives the final classification decision",
], font_size=14, color=TEXT_MED)

# Arrows between cards
for arrow_left in [Inches(4.35), Inches(8.55)]:
    add_text(slide6, arrow_left, Inches(3.5), Inches(0.4), Inches(0.5),
             "→", font_size=36, color=ACCENT, bold=True, alignment=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════
# SLIDE 7 – Key Advantages & Efficiency
# ════════════════════════════════════════════════════════════════════
slide7 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide7, BG_LIGHT)

add_text(slide7, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7),
         "Key Advantages & Efficiency",
         font_size=34, color=TEXT_DARK, bold=True)
add_shape(slide7, Inches(0.6), Inches(0.95), Inches(3), Inches(0.05), ACCENT)

# Card 1 – Compute-Light
add_shape(slide7, Inches(0.4), Inches(1.4), Inches(6.2), Inches(2.5), BG_CARD, ACCENT2)
add_text(slide7, Inches(0.7), Inches(1.55), Inches(5.5), Inches(0.5),
         "⚡  Compute-Light", font_size=20, color=ACCENT, bold=True)
add_bullet_text(slide7, Inches(0.7), Inches(2.1), Inches(5.7), Inches(1.7), [
    "Completely bypasses expensive model training",
    "No backpropagation or gradient updates required",
    "Runs on minimal hardware — no GPU needed for inference",
], font_size=14, color=TEXT_MED)

# Card 2 – High Speed
add_shape(slide7, Inches(6.8), Inches(1.4), Inches(6.2), Inches(2.5), BG_CARD, ACCENT2)
add_text(slide7, Inches(7.1), Inches(1.55), Inches(5.5), Inches(0.5),
         "🚀  High Speed", font_size=20, color=ACCENT, bold=True)
add_bullet_text(slide7, Inches(7.1), Inches(2.1), Inches(5.7), Inches(1.7), [
    "Relies on precomputed embeddings for instant lookup",
    "Highly optimized nearest-neighbor search algorithms",
    "Rapid inference suitable for real-time applications",
], font_size=14, color=TEXT_MED)

# Card 3 – Granular Output
add_shape(slide7, Inches(0.4), Inches(4.2), Inches(6.2), Inches(2.5), BG_CARD, ACCENT2)
add_text(slide7, Inches(0.7), Inches(4.35), Inches(5.5), Inches(0.5),
         "🔬  Granular Output", font_size=20, color=ACCENT, bold=True)
add_bullet_text(slide7, Inches(0.7), Inches(4.9), Inches(5.7), Inches(1.7), [
    "Supports fine-grained taxonomy classification",
    "Matches input to specifically labeled clusters",
    "Inherent multi-class support without architectural changes",
], font_size=14, color=TEXT_MED)

# Card 4 – Instant Adaptability
add_shape(slide7, Inches(6.8), Inches(4.2), Inches(6.2), Inches(2.5), BG_CARD, ACCENT2)
add_text(slide7, Inches(7.1), Inches(4.35), Inches(5.5), Inches(0.5),
         "🔄  Instant Adaptability", font_size=20, color=ACCENT, bold=True)
add_bullet_text(slide7, Inches(7.1), Inches(4.9), Inches(5.7), Inches(1.7), [
    "Updated in real-time by adding new examples to the vector DB",
    "Zero downtime required for system updates",
    "New manipulation patterns recognized immediately after insertion",
], font_size=14, color=TEXT_MED)

# ════════════════════════════════════════════════════════════════════
# SLIDE 8 – Tradeoffs & Structural Limitations
# ════════════════════════════════════════════════════════════════════
slide8 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide8, BG_LIGHT)

add_text(slide8, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7),
         "Tradeoffs & Structural Limitations",
         font_size=34, color=TEXT_DARK, bold=True)
add_shape(slide8, Inches(0.6), Inches(0.95), Inches(3), Inches(0.05), ORANGE)

# Card 1 – Embedding Dependency
add_shape(slide8, Inches(0.4), Inches(1.5), Inches(3.9), Inches(5.0), BG_CARD, ORANGE)
add_shape(slide8, Inches(0.6), Inches(1.7), Inches(3.2), Inches(0.45), ORANGE)
add_text(slide8, Inches(0.65), Inches(1.72), Inches(3.1), Inches(0.4),
         "EMBEDDING DEPENDENCY", font_size=13, color=RGBColor(0xFF, 0xFF, 0xFF), bold=True, alignment=PP_ALIGN.CENTER)
add_text(slide8, Inches(0.7), Inches(2.4), Inches(3.3), Inches(0.5),
         "⚠️  Quality Bottleneck", font_size=20, color=ORANGE, bold=True)
add_bullet_text(slide8, Inches(0.7), Inches(3.0), Inches(3.4), Inches(3.2), [
    "Entire system accuracy is bottlenecked by embedding quality",
    "If embeddings fail to capture pragmatic nuance, retrieval fails",
    "Choice of embedding model is a critical design decision",
], font_size=14, color=TEXT_MED, bullet_color=ORANGE)

# Card 2 – Vulnerability to Novelty
add_shape(slide8, Inches(4.6), Inches(1.5), Inches(3.9), Inches(5.0), BG_CARD, ORANGE)
add_shape(slide8, Inches(4.8), Inches(1.7), Inches(3.4), Inches(0.45), ORANGE)
add_text(slide8, Inches(4.85), Inches(1.72), Inches(3.3), Inches(0.4),
         "VULNERABILITY TO NOVELTY", font_size=13, color=RGBColor(0xFF, 0xFF, 0xFF), bold=True, alignment=PP_ALIGN.CENTER)
add_text(slide8, Inches(4.9), Inches(2.4), Inches(3.3), Inches(0.5),
         "🆕  Out-of-Domain Weakness", font_size=20, color=ORANGE, bold=True)
add_bullet_text(slide8, Inches(4.9), Inches(3.0), Inches(3.4), Inches(3.2), [
    "Easily fooled by out-of-domain phrasing patterns",
    "Highly novel sentence structures won't map near stored cases",
    "Requires continuous exemplar expansion for coverage",
], font_size=14, color=TEXT_MED, bullet_color=ORANGE)

# Card 3 – Semantic Blind Spots
add_shape(slide8, Inches(8.8), Inches(1.5), Inches(3.9), Inches(5.0), BG_CARD, ORANGE)
add_shape(slide8, Inches(9.0), Inches(1.7), Inches(3.2), Inches(0.45), ORANGE)
add_text(slide8, Inches(9.05), Inches(1.72), Inches(3.1), Inches(0.4),
         "SEMANTIC BLIND SPOTS", font_size=13, color=RGBColor(0xFF, 0xFF, 0xFF), bold=True, alignment=PP_ALIGN.CENTER)
add_text(slide8, Inches(9.1), Inches(2.4), Inches(3.3), Inches(0.5),
         "👁️  False Positive Risk", font_size=20, color=ORANGE, bold=True)
add_bullet_text(slide8, Inches(9.1), Inches(3.0), Inches(3.4), Inches(3.2), [
    "Prioritizes semantic closeness over explicit logical reasoning",
    "May misclassify benign text sharing vocabulary with manipulative data",
    "Lacks ability to reason about context or speaker intent",
], font_size=14, color=TEXT_MED, bullet_color=ORANGE)

# ════════════════════════════════════════════════════════════════════
# SLIDE 9 – Response Correction Layer (New)
# ════════════════════════════════════════════════════════════════════
slide9 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide9, BG_LIGHT)

add_text(slide9, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7),
         "New Response Correction Layer",
         font_size=34, color=TEXT_DARK, bold=True)
add_shape(slide9, Inches(0.6), Inches(0.95), Inches(3.2), Inches(0.05), ACCENT)

# Left side: mechanism summary
add_shape(slide9, Inches(0.4), Inches(1.5), Inches(5.9), Inches(5.7), BG_CARD, ACCENT2)
add_text(slide9, Inches(0.7), Inches(1.7), Inches(5.3), Inches(0.5),
         "How It Works", font_size=22, color=ACCENT, bold=True)
add_bullet_text(slide9, Inches(0.7), Inches(2.25), Inches(5.4), Inches(2.8), [
    "After classification, pipeline generates a normal assistant reply",
    "If manipulative=1, a hidden supervisor instruction is injected",
    "Instruction: politely acknowledge, refuse manipulative premise, pivot safe",
    "If manipulative=0, corrected response remains equivalent to base response",
], font_size=14, color=TEXT_MED)

add_shape(slide9, Inches(0.7), Inches(5.35), Inches(5.3), Inches(1.55), RGBColor(0xF0, 0xF0, 0xF5), ACCENT2)
add_text(slide9, Inches(0.9), Inches(5.5), Inches(4.9), Inches(1.2),
         "Implementation path:\n"
         "pipeline.py -> call_llm_with_intervention(...)\n"
         "llm_client.py injects system supervisor message",
         font_size=13, color=TEXT_MED)

# Right side: working example
add_shape(slide9, Inches(6.6), Inches(1.5), Inches(6.3), Inches(5.7), BG_CARD, ACCENT2)
add_text(slide9, Inches(6.9), Inches(1.7), Inches(5.7), Inches(0.5),
         "Working Example (from predictions.csv)", font_size=22, color=ACCENT, bold=True)
add_text(slide9, Inches(6.9), Inches(2.2), Inches(5.7), Inches(0.45),
         "Dialogue ID: 85514416  |  manipulative=1  |  confidence=0.90",
         font_size=13, color=TEXT_MED)

add_shape(slide9, Inches(6.9), Inches(2.7), Inches(5.7), Inches(1.7), RGBColor(0xF8, 0xFB, 0xFF), RGBColor(0xC6, 0xDB, 0xF0))
add_text(slide9, Inches(7.1), Inches(2.85), Inches(5.3), Inches(1.35),
         "Base assistant response:\n"
         "\"Person1, I would advise against consuming any more tea until the\n"
         "situation is resolved safely.\"",
         font_size=12, color=TEXT_DARK)

add_shape(slide9, Inches(6.9), Inches(4.55), Inches(5.7), Inches(2.3), RGBColor(0xF2, 0xFA, 0xF4), RGBColor(0xB8, 0xDD, 0xC1))
add_text(slide9, Inches(7.1), Inches(4.7), Inches(5.3), Inches(2.0),
         "Corrected assistant response:\n"
         "\"I understand your concern, but I must emphasize that using Deadly\n"
         "Night Shade is extremely dangerous and not a solution to any problem.\n"
         "Let's focus on a safe and healthy way to address this.\"",
         font_size=12, color=GREEN)

# Outcome banner
add_shape(slide9, Inches(0.4), Inches(7.0), Inches(12.5), Inches(0.35), BG_CARD, ACCENT)
add_text(slide9, Inches(0.5), Inches(7.01), Inches(12.2), Inches(0.3),
         "Outcome: Converts potentially compliant/neutral replies into safety-aligned refusals for manipulative interactions.",
         font_size=12, color=TEXT_MED, alignment=PP_ALIGN.CENTER)

# ── Save ──
output_path = r"c:\Users\gopic\OneDrive\Desktop\NLP\iap_pipeline\IAP_Pipeline_Presentation.pptx"
prs.save(output_path)
print(f"Presentation saved to: {output_path}")
