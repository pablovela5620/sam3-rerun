import tyro

from sam3d_body.api.rr_demo import Sam3DBodyConfig, main

if __name__ == "__main__":
    main(tyro.cli(Sam3DBodyConfig))
