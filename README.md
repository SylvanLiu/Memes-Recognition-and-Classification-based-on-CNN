# Social-Images(Memes) Recognition and Classification Based on Convolutional Neural Network.

## Abstract

With the popularity of ‘Internet Social Culture’ in China, specific images that used to be made, generally with hilarious content, and spread rapidly in subcultures like post bars and other kinds of online forums, have also been gradually accepted and then appreciated by the masses. Furthermore, these ‘Social Function Pictures’ are called ‘memes’ today, and largely used in daily communication.

Most material of them come from capturing amusing moments of some characters in the film and television works or someone in our life, and further processing like secondary creations by fusing with linear hand-painted styles or adding subtitles will add new energy and put the soul into these ‘Social Function Pictures’.

The biggest difference between them and ‘emojis’, and other pictures is, these ‘Social Function Pictures’ are spontaneously produced by net friends, then spread via the Internet without external interventions. They carry exceedingly complicated information and connotation by simple static content, hold the sensory characteristics which text language doesn’t have. They can make our chat more three-dimensional, make users to be more accurately and conveniently when expressing their inner thoughts and real situations, and it is far easier for receivers to understand and feel the empathy.

The appearance of ‘Social Function Pictures’ perfectly meets the demand of the Internet era, so their content and scale have been exponentially increasing in recent years. An accompanying problem is, we usually need to pick out one or more suitable options from a large number of images with a given situation. However, these suitable options are not adjacent for most of the time, there will be a lot of distractions between them, for example images we saved from other apps.

Notwithstanding we can quickly comprehend and integrate extremely abstract content in enigmatic approaches that no single computer can truly compare, our limitations are also very obvious, it is totally impossible for most of us to exert calculation on the mass data by giving an appointed mathematical logic structure and formula, even if it is quite simple and we can assume that we will never feel tired and bored about this, subsequently come out the exact result set.

The more frequently we use ‘Social Function Pictures’ in our daily life, the eagerer are the needs for us to develop an agile and efficient method that can retrieve the deep-seated semantics of them by the subjective wants of people in specific situations.

## 1. Data Preparation and Pretest Analysis

In order to get enough data for this dissertation in a faster and more convenient way, I have asked two best friends of mine for some indispensable help. I knew that they have a great quantity of images inside their computers and mobile phones, which have been stored locally as cache files but are never wiped, from the variety of mainstream social media and social software. Besides this, most of their images are the kinds I need for this dissertation.

Therefore, I copied all files from their devices, with sincere gratitude, as raw material of my following exploration and experiment. As a matter of fact, we used to share these amusing images with each other , not only for fun, we all consider that it is really crucial to refresh our ‘stock’ timely in order to help us not to be knocked out in contemporary era that everyone is facing ‘Internet of Everything’, the brand new hot issues and topics will burst into your eyes in the next second. And this is no longer the technology only need to be mastered by young people, though we are not willing to admit that, it is a real happened phenomenon ,that currently, conversation will get exceedingly awkward and weird if we stop using any ‘Social Function Pictures’ inside it.(and I reckon that in this situation, emojis should be considered as ‘Social Function Pictures’ too.)

Before embarking on the main task, I had tried to dig some common attributes by utilizing simple big data analysis on those images. During I was doing this, one thing deserves attention was that there would exist a certain amount of images were not in their right extensions, it would incur unexpected errors in our following operations. To solve this, we can discriminate their real formats by opening the file in binary, then read a specific serial number of each file, the four sets of numbers in the header of images files can manifest their actual methods of coding. All kinds of formats appeared in the entire data set, and their proportions, are listed below, with their specific serial numbers in decimal and Hexadecimal forms. (Items which have been marked gray are abandoned due to they are useless at all, meanwhile, we only need to focus on these four formats are not in gray which contain all possibilities we will meet in the future.) 

| Type Name  | Hex Code | Decimal Code |
| ------------- | ------------- | ------------- |
| Standard JPEG/JFIF Format(57.92%) | 0xFF 0xD8 0xFF 0xE0 |  [255, 216, 255, 224] |
|PNG File (21.75%)|0x89 0x50 0x4E 0x47|[137, 80, 78, 71]|
|Standard JPEG/EXIF Format (12.06%)|0xFF 0xD8 0xFF 0xE1|[255, 216, 255, 225]|
|GIF Format (8.21%)|0x47 0x49 0x46 0x38|[71, 73, 70, 56]|
|Samsung Non-Standard JPEG Format (0.20%)|0xFF 0xD8 0xFF 0xDB|[255, 216, 255, 219]|
|HyperText Markup Language 3 Format (0.03%)|0x3C 0x21 0x44 0x4F|[60, 33, 68, 79] |
|Other Non-Standard JPEG Formats (0.03%)|0xFF 0xD8 0xFF 0xC0|[255, 216, 255, 192]|
|Ditto|0xFF 0xD8 0xFF 0xE2|[255, 216, 255, 226]|
|Ditto|0xFF 0xD8 0xFF 0xE9|[255, 216, 255, 233]|
|Ditto|0xFF 0xD8 0xFF 0xED|[255, 216, 255, 237]|
|Ditto|0xFF 0xD8 0xFF 0xFE|[255, 216, 255, 254]|
|Other 16 Types of Equivocal Formats (0.00%)|	N/A	| N/A |

### 1.1 	Graph [1]
