from functools import partial
import os

import torch
import numpy as np
import gradio as gr
import gdown

from load import load_model, load_json
from load import load_unit_motion_embs_splits, load_keyids_splits


EXAMPLES = [
    "A person is walking in a circle",
    "A person is jumping rope",
    "Someone is doing a backflip",
    "A person is doing a moonwalk",
    "A person walks forward and then turns back",
    "Picking up an object",
    "A person is swimming in the sea",
    "A human is squatting",
    "Someone is jumping with one foot",
    "A person is chopping vegetables",
    "Someone walks backward",
    "Somebody is ascending a staircase",
    "A person is sitting down",
    "A person is taking the stairs",
    "Someone is doing jumping jacks",
    "The person walked forward and is picking up his toolbox",
    "The person angrily punching the air."
]

# Show closest text in the training


# css to make videos look nice
CSS = """
video {
    position: relative;
    margin: 0;
    box-shadow: var(--block-shadow);
    border-width: var(--block-border-width);
    border-color: var(--block-border-color);
    border-radius: var(--block-radius);
    background: var(--block-background-fill);
    width: 100%;
    line-height: var(--line-sm);
}
"""


def humanml3d_keyid_to_babel_rendered_url(h3d_index, amass_to_babel, keyid):
    # Don't show the mirrored version of HumanMl3D
    if "M" in keyid:
        return None

    dico = h3d_index[keyid]
    path = dico["path"]

    # HumanAct12 motions are not rendered online
    # so we skip them for now
    if "humanact12" in path:
        return None

    # This motion is not rendered in BABEL
    # so we skip them for now
    if path not in amass_to_babel:
        return None

    babel_id = amass_to_babel[path].zfill(6)
    url = f"https://babel-renders.s3.eu-central-1.amazonaws.com/{babel_id}.mp4"

    # For the demo, we retrieve from the first annotation only
    ann = dico["annotations"][0]
    start = ann["start"]
    end = ann["end"]
    text = ann["text"]

    data = {
        "url": url,
        "start": start,
        "end": end,
        "text": text,
        "keyid": keyid,
        "babel_id": babel_id
    }

    return data


def retrieve(model, keyid_to_url, all_unit_motion_embs, all_keyids, text, splits=["test"], nmax=8):
    unit_motion_embs = torch.cat([all_unit_motion_embs[s] for s in splits])
    keyids = np.concatenate([all_keyids[s] for s in splits])

    scores = model.compute_scores(text, unit_embs=unit_motion_embs)

    sorted_idxs = np.argsort(-scores)
    best_keyids = keyids[sorted_idxs]
    best_scores = scores[sorted_idxs]

    datas = []
    for keyid, score in zip(best_keyids, best_scores):
        if len(datas) == nmax:
            break

        data = keyid_to_url(keyid)
        if data is None:
            continue
        data["score"] = round(float(score), 2)
        datas.append(data)
    return datas


# HTML component
def get_video_html(url, video_id, start=None, end=None, score=None, width=350, height=350):
    trim = ""
    if start is not None:
        if end is not None:
            trim = f"#t={start},{end}"
        else:
            trim = f"#t={start}"

    score_t = ""
    if score is not None:
        score_t = f'title="Score = {score}"'

    video_html = f'''
<video preload="auto" muted playsinline onpause="this.load()"
autoplay loop disablepictureinpicture id="{video_id}" width="{width}" height="{height}" {score_t}>
  <source src="{url}{trim}" type="video/mp4">
  Your browser does not support the video tag.
</video>
'''
    return video_html


def retrive_component(retrieve_function, text, splits, nvids, n_component=16):
    # cannot produce more than n_compoenent
    nvids = min(nvids, n_component)
    if not splits:
        return [None for _ in range(n_component)]

    splits_l = [x.lower() for x in splits]
    datas = retrieve_function(text, splits=splits_l, nmax=nvids)
    htmls = [
        get_video_html(
            url["url"], idx, start=url["start"],
            end=url["end"], score=url["score"]
        )
        for idx, url in enumerate(datas)
    ]
    # get n_component exactly if asked less
    # pad with dummy blocks
    htmls = htmls + [None for _ in range(max(0, n_component-nvids))]
    return htmls


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # LOADING
    model = load_model(device)
    splits = ["train", "val", "test"]
    all_unit_motion_embs = load_unit_motion_embs_splits(splits, device)
    all_keyids = load_keyids_splits(splits)

    h3d_index = load_json("amass-annotations/humanml3d.json")
    amass_to_babel = load_json("amass-annotations/amass_to_babel.json")

    keyid_to_url = partial(humanml3d_keyid_to_babel_rendered_url, h3d_index, amass_to_babel)
    retrieve_function = partial(retrieve, model, keyid_to_url, all_unit_motion_embs, all_keyids)

    # DEMO
    theme = gr.themes.Default(primary_hue="blue", secondary_hue="gray")
    retrive_and_show = partial(retrive_component, retrieve_function)

    default_text = "A person is "

    with gr.Blocks(css=CSS, theme=theme) as demo:
        title = "<h1 style='text-align: center'>TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis </h1>"
        gr.Markdown(title)

        authors = """
        <h2 style='text-align: center'>
        <a href="https://mathis.petrovich.fr" target="_blank"><nobr>Mathis Petrovich</nobr></a> &emsp;
        <a href="https://ps.is.mpg.de/~black" target="_blank"><nobr>Michael J. Black</nobr></a> &emsp;
	<a href="https://imagine.enpc.fr/~varolg" target="_blank"><nobr>G&uumll Varol</nobr></a>
	</h2>
        """
        gr.Markdown(authors)

        conf = """
        <h2 style='text-align: center'>
	<nobr>arXiv 2023</nobr>
	</h2>
        """
        gr.Markdown(conf)

        videos = []

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Column(scale=2):
                    text = gr.Textbox(placeholder="Type in natural language, the motion to retrieve",
                                      show_label=True, label="Text prompt", value=default_text)
                with gr.Column(scale=1):
                    btn = gr.Button("Retrieve", variant='primary')
                    clear = gr.Button("Clear", variant='secondary')

                with gr.Row():
                    with gr.Column(scale=1):
                        splits = gr.Dropdown(["Train", "Val", "Test"],
                                             value=["Test"], multiselect=True, label="Splits",
                                             info="HumanML3D data used for the motion database")
                    with gr.Column(scale=1):
                        nvideo_slider = gr.Slider(minimum=4, maximum=16, step=4, value=8, label="Number of videos")
            with gr.Column(scale=2):
                examples = gr.Examples(examples=EXAMPLES, inputs=text, examples_per_page=15)

        i = -1
        # should indent
        for _ in range(4):
            with gr.Row():
                for _ in range(4):
                    i += 1
                    with gr.Column():
                        video = gr.HTML()
                        videos.append(video)

        def check_error(splits):
            if not splits:
                raise gr.Error("At least one split should be selected!")
            return splits

        btn.click(fn=retrive_and_show, inputs=[text, splits, nvideo_slider], outputs=videos).then(
            fn=check_error, inputs=splits
        )

        text.submit(fn=retrive_and_show, inputs=[text, splits, nvideo_slider], outputs=videos).then(
            fn=check_error, inputs=splits
        )

        def keep_test(splits):
            if len(splits) == 0:
                return ["Test"]
            return splits

        def clear_videos():
            return [None for x in range(16)] + [default_text]

        clear.click(fn=clear_videos, outputs=videos + [text])
    demo.launch()


def prepare():
    if not os.path.exists("data"):
        gdown.download_folder("https://drive.google.com/drive/folders/1MgPFgHZ28AMd01M1tJ7YW_1-ut3-4j08", use_cookies=False)


if __name__ == "__main__":
    prepare()
    main()

# new
# A person is walking slowly
