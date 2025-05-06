import os
import json
import pickle
import argparse
from loguru import logger
from tqdm.auto import tqdm

from src.vsg.pipelines import set_pipeline
from src.datasets.utils import draw_nx_sg
from src.vsg.handlers.utils import triplets2graph, triplets2graph_gpt


def main(args):
    pipeline = set_pipeline[args.sgg_method]()

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "sg_pickle"), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "sg_pickle_viz"), exist_ok=True)

    with open(os.path.join(args.output_folder, "captions.json"), "wt") as captions_file:
        for image_path in tqdm(sorted(os.listdir(args.image_folder), key=lambda i: int(i.split("_")[1]))):
            logger.info(image_path)

            caption = pipeline.image2text(os.path.join(args.image_folder, image_path))
            captions_file.write(json.dumps({"image": image_path, "caption": caption}) + "\n")

            if "gpt" in args.sgg_method:
                nx_sg = triplets2graph_gpt(pipeline.text2triplets(caption))
            else:
                nx_sg = triplets2graph(pipeline.text2triplets(caption))
            with open(os.path.join(args.output_folder, 
                                   "sg_pickle", 
                                   image_path.split(".")[0] + ".pkl"), 
                                       "wb") as g_file:
                pickle.dump(nx_sg, g_file)

            vis_save_path = os.path.join(args.output_folder, "sg_pickle_viz", image_path)
            draw_nx_sg(nx_sg, vis_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sgg_method', help='one of gpt|nvila_factual|nvila_gpt', required=True,
                        choices=["gpt", "nvila_factual", "nvila_gpt"])
    parser.add_argument('--image_folder', help='path to images', required=True)
    parser.add_argument('--output_folder', help='path to save itermidiate files', required=True)
    args = parser.parse_args()

    main(args)
