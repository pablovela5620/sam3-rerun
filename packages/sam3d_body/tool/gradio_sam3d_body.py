from sam3d_body.gradio_ui.sam3d_body_ui import main

if __name__ == "__main__":
    demo = main()
    demo.launch(ssr_mode=False)
