
# NeuralTextures

This is repository with inference code for paper **"StylePeople: A Generative Model of Fullbody Human Avatars"** (CVPR21).
This code is for the part of the paper describing video-based avatars. For inference of generative neural textures model refer to [this repository](https://github.com/saic-vul/style-people).

## Getting started
### Data
To use this repository you first need to download model checkpoints and some auxiliary files.

* Download the archive with data from [Google Drive](https://drive.google.com/drive/folders/1YcY3WtCGyq6c0cZIcCG1rll7HGZb_JXc?usp=sharing) and unpack it into `NeuralTextures/data/`. It contains:
	* checkpoints for generative model and encoder network (`data/checkpoint`)
	* SMPL-X parameters for samples from *AzurePeople* dataset to run inference script on (`data/smplx_dicts`)
	* Some auxiliary data (`data/uv_render` and `data/*.npy`)
* Download SMPL-X models (`SMPLX_{MALE,FEMALE,NEUTRAL}.pkl`) from [SMPL-X project page](https://smpl-x.is.tue.mpg.de/) and move them to `data/smplx/`

### Docker
The easiest way to build an environment for this repository is to use docker image. To build it, make the following steps:
1. Build the image with the following command:
```
bash docker/build.sh
```
2. Start a container:
```
bash docker/run.sh
```
It mounts root directory of the host system to `/mounted/` inside docker and sets cloned repository path as a starting directory.

3. **Inside the container** install `minimal_pytorch_rasterizer`. (Unfortunately, docker fails to install it during image building)
```
pip install git+https://github.com/rmbashirov/minimal_pytorch_rasterizer
```
4. *(Optional)* You can then commit changes to the image so that you don't need to install  `minimal_pytorch_rasterizer` for every new container. See [docker documentation](https://docs.docker.com/engine/reference/commandline/commit/).

## Usage   
For now the only scenario in this repository involves rendering an image of a person from *AzurePeople* dataset with giver SMPL-X parameters.

Example:
```
python render_azure_person.py --person_id=04 --smplx_dict_path=data/smplx_dicts/04.pkl --out_path=data/results/
```
will render a person with `id='04'` with SMPL-X parameters from `data/smplx_dicts/04.pkl` and save resulting images to `data/results/04`.

For ids of all 56 people consult [this table](assets/dataset_lookup.png)
