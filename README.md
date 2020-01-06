# pero-enhance

Tool for text-guided textual document scan quality enhancement. The method works on lines of text that can be input through a PAGE XML or detected automatically by a buil-in OCR. By using text input along with the image, the results can be correctly readable even with parts of the original text missing or severly degraded in the source image. The tool includes functionality for cropping the text lines, processing them with our provided  models for either text enhancement and inpainting, and for blending the enhanced text lines back into the source document image. We currently provide models for OCR and enhancement of czech newspapers optimized for low-quality scans from micro-films.

<img src="images/orig.png" height="256"> <img src="images/enhanced_correct.png" height="256">

The method is based on Generative Adversarial Neural Networks (GAN) that are trained on pairs of good quality and bad quality text document examples. The architecture includes convolutional encoder and decoder for repairing the visual quality of the text line image and transformer module with attention mechanism that aligns input text string to the image to provide more information for the decoder.

## Installation
Clone the repository and add the pero_enhance and pero_ocr package to your `PYTHONPATH`:
```
clone https://github.com/DCGM/pero-enhance.git
export PYTHONPATH=/abs/path/to/repo/pero-enhance:/abs/path/to/repo/pero-enhance/pero-ocr:$PYTHONPATH
```
Install other dependencies:
```
pip install -r requirements.txt
```
Before processing a document, you need to download configuration and models needed for enhancement and OCR: 
```
python ./model/download_models.py
```
By default, models for czech newspaper is downloaded. Other models can be found in the table below. The list will be updated as we prepare more models.

## Usage
### Demo
The enhancement uses document information in PAGE XML format. Folder of images can be enhanced by running following:
```
python repair_page.py -i ./example/ -x ./example/ -o /path/to/outputs
```
Alternatively, you can run interactive demo by running the following:
```
python demo.py -i ./example/82f4ac84-6f1e-43ba-b1d5-e2b28d69508d.jpg -x ./example/82f4ac84-6f1e-43ba-b1d5-e2b28d69508d.xml
```
In case of XML missing or the path to XML file/folder not specified, automatic text detection and OCR is done using the pero `PageParser`. The default path to the downloaded enhancement and OCR models can be modified using `-r /path/to/enhancement/json`and `-p /path/to/ocr/config`, respectively. 

### EngineRepairCNN class
In your code, you can also use directly the EngineRepairCNN to enhance invidiual lines or page defined by pero.layout class.
```
import repair_engine
enhancer = repair_engine.EngineRepairCNN(path/to/repair_engine.json)

enhanced_textline_image = enhancer.repair_line(textline_image, transcription_string)

inpainted_textline_image = enhancer.inpaint_line(textline_image, transcription_string)

enhanced_page_image = enhancer.enhance_page(page_img, page_layout)
```
To inpaint a part of the textline, the model expects the according part of the textline image to be blacked-out.

## Models

 | OCR Model | Enhancement Model | Target data |
 | --- | --- | --- |
 | [ocr_LN_2019-12-18.zip](http://www.fit.vutbr.cz/~ihradis/pero-models/ocr_LN_2019-12-18.zip) | [enhance_LN_2019-12-18.zip](http://www.fit.vutbr.cz/~ihradis/pero-models/enhance_LN_2019-12-18.zip) | Czech newspaper (Lidové Noviny) |
 | More to come | | |
