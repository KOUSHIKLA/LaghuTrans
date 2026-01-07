# -*- coding: utf-8 -*-
import time
import gradio as gr
from inference_core import process_pipeline

def translate_text(text):
    if not text.strip():
        return "", "0.00 sec"
    start = time.time()
    output = process_pipeline(text)
    end = time.time()
    return output, f"{round(end - start, 2)} sec"

# Custom CSS for better styling
css = """
.gradio-container {
    max-width: 1400px !important;
}

#title {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 0.5em;
}

#subtitle {
    text-align: center;
    color: #666;
    font-size: 1.1em;
    margin-bottom: 2em;
}

.translate-btn {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-size: 1.1em !important;
    padding: 12px !important;
}
"""

# Create interface with Blocks
with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    
    gr.Markdown("<h1 id='title'>LaghuTrans: English to Hindi Neural Machine Translation</h1>")
    gr.Markdown("<p id='subtitle'>Specialized for Agriculture & Judicial Domains | High-Quality Domain-Specific Translation</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="English Input",
                placeholder="Enter English text here...\n\nTry domain-specific sentences from Agriculture or Judicial fields for best results.",
                lines=8,
                show_label=True
            )
            
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                translate_btn = gr.Button("Translate", variant="primary", scale=3, elem_classes="translate-btn")
        
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Hindi Output",
                lines=8,
                show_label=True,
                interactive=False
            )
            
            time_box = gr.Textbox(
                label="Inference Time",
                interactive=False,
                lines=1
            )
    
    # Examples section
    gr.Markdown("### Try These Example Sentences:")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**Agriculture Examples:**")
            gr.Examples(
                examples=[
                    ["The farmer planted rice in the paddy fields during the monsoon season."],
                    ["Organic farming reduces the use of chemical fertilizers and pesticides."],
                    ["Crop rotation helps maintain soil fertility and prevent pest infestations."],
                    ["Integrated pest management combines biological, cultural, and chemical methods to control crop diseases."]
                ],
                inputs=input_text,
                label=None
            )
        
        with gr.Column(scale=1):
            gr.Markdown("**Judicial Examples:**")
            gr.Examples(
                examples=[
                    ["The accused has the right to remain silent during police interrogation."],
                    ["The judge adjourned the hearing to next Monday."],
                    ["The lawyer filed a petition in the High Court."],
                    ["The appellate court upheld the lower court's verdict and dismissed the appeal."]
                ],
                inputs=input_text,
                label=None
            )
    
    # Information section with updated performance metrics
    with gr.Accordion("About LaghuTrans Model", open=False):
        gr.Markdown("""
        ### Model Information
        - **Model Name**: LaghuTrans (Lightweight Translation)
        - **Architecture**: 150M parameter Neural Machine Translation model
        - **Parameters**: 150M | **FLOPs**: 188.62G | **MACs**: 94.23G
        - **Training**: Specialized for Agriculture & Judicial domains
        - **Languages**: English to Hindi
        
        ### Performance Metrics
        LaghuTrans achieves strong translation quality with BLEU scores of 34-44 and COMET scores of 82-89 across benchmark datasets, while offering 2-4x faster inference than larger baseline models.
        
        **Benchmark Scores:**
        - **BLEU**: 34.83 (IN-22 Gen) | 44.09 (Flores-200) | 33.22 (IN-22 Conv)
        - **ChrF++**: 52.92 (IN-22 Gen) | 61.67 (Flores-200) | 52.94 (IN-22 Conv)
        - **COMET**: 86.3 (IN-22 Gen) | 82.1 (Flores-200) | 88.7 (IN-22 Conv)
        - **Inference Speed**: 29-69 seconds (dataset dependent) - significantly faster than 418M parameter models
        
        ### Why LaghuTrans?
        - High-quality domain-specific translations
        - Efficient and fast inference
        - Optimized for professional use in agriculture and legal sectors
        - Balanced trade-off between accuracy and computational efficiency
        - Outperforms many larger models on domain-specific content
        
        ### Best Use Cases
        - Agricultural documents and reports
        - Legal and judicial content
        - Technical documentation in these domains
        - Professional translations requiring domain expertise
        - Government and institutional document translation
        
        ### Tips for Best Results
        - Use complete, grammatically correct sentences
        - The model performs best on agriculture and judicial content
        - Avoid very informal or colloquial language
        - For technical terms, ensure proper context is provided
        """)
    
    # Event handlers
    translate_btn.click(
        fn=translate_text,
        inputs=input_text,
        outputs=[output_text, time_box]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", ""),
        inputs=None,
        outputs=[input_text, output_text, time_box]
    )
    
    # Also allow Enter key to translate
    input_text.submit(
        fn=translate_text,
        inputs=input_text,
        outputs=[output_text, time_box]
    )

# Launch
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)



































#import time
#import gradio as gr
#from inference_core import process_pipeline
#
#
#def translate_text(text):
#    if not text.strip():
#        return "", "0.00 sec"
#
#    start = time.time()
#    output = process_pipeline(text)
#    end = time.time()
#
#    return output, f"{round(end - start, 2)} sec"
#
#
#with gr.Blocks(title="English → Hindi MT") as demo:
#    gr.Markdown("## 🇮🇳 English → Hindi Machine Translation")
#
#    with gr.Row():
#        with gr.Column():
#            input_text = gr.Textbox(
#                label="English Input",
#                placeholder="Enter English sentence",
#                lines=4
#            )
#            translate_btn = gr.Button("Translate")
#
#        with gr.Column():
#            output_text = gr.Textbox(
#                label="Hindi Output",
#                lines=4
#            )
#            time_box = gr.Textbox(
#                label="Inference Time",
#                interactive=False
#            )
#
#    translate_btn.click(
#        translate_text,
#        inputs=input_text,
#        outputs=[output_text, time_box]
#    )
#
#demo.launch(
#    server_name="0.0.0.0",
#    server_port=7860,
#    share=False
#)
