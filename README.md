# Social-Images(Memes) Recognition and Classification Based on Convolutional Neural Network.

The folder with name 'labelled_Data' contains a partial dataset of self-labelled memes only for exhibition, which were picked from an around-300GB, 600 thousand unlabled raw memes(If you need that for having some fun, just feel free to contact me);
The folder with name 'Img4Test' contains the samples for testing the local python scripts which process and pre-process local memes.

The project for my Bachelor's dissertation, tries to recognise and classify the 'Social Function Pictures' (memes) by their emotion and content. (Includes a self-labelled memes data set). It was proposed merely for my personal interest, partially implemented by local python scripts for data processing and pre-processing, and then deployed on Aliyun online PAI machine learning platform with online configuration files for training and predicting.


#### Key word: Computer Vision; Image Processing; Deep Learning; Convolutional Neural Network; Emotion Recognition ; Affective computing.

## Abstract

With the popularity of ‘Internet Social Culture’ in China, specific images that used to be made, generally with hilarious content, and spread rapidly in subcultures like post bars and other kinds of online forums, have also been gradually accepted and then appreciated by the masses. Furthermore, these ‘Social Function Pictures’ are called ‘memes’ today, and largely used in daily communication.

Most material of them come from capturing amusing moments of some characters in the film and television works or someone in our life, and further processing like secondary creations by fusing with linear hand-painted styles or adding subtitles will add new energy and put the soul into these ‘Social Function Pictures’.

The biggest difference between them and ‘emojis’, and other pictures is, these ‘Social Function Pictures’ are spontaneously produced by net friends, then spread via the Internet without external interventions. They carry exceedingly complicated information and connotation by simple static content, hold the sensory characteristics which text language doesn’t have. They can make our chat more three-dimensional, make users to be more accurately and conveniently when expressing their inner thoughts and real situations, and it is far easier for receivers to understand and feel the empathy.

The appearance of ‘Social Function Pictures’ perfectly meets the demand of the Internet era, so their content and scale have been exponentially increasing in recent years. An accompanying problem is, we usually need to pick out one or more suitable options from a large number of images with a given situation. However, these suitable options are not adjacent for most of the time, there will be a lot of distractions between them, for example images we saved from other apps.

Notwithstanding we can quickly comprehend and integrate extremely abstract content in enigmatic approaches that no single computer can truly compare, our limitations are also very obvious, it is totally impossible for most of us to exert calculation on the mass data by giving an appointed mathematical logic structure and formula, even if it is quite simple and we can assume that we will never feel tired and bored about this, subsequently come out the exact result set.

The more frequently we use ‘Social Function Pictures’ in our daily life, the eagerer are the needs for us to develop an agile and efficient method that can retrieve the deep-seated semantics of them by the subjective wants of people in specific situations.

## Data Preparation and Pretest Analysis

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

<p align="center">
  <img width="500" height="500" src="https://github.com/SylvanLiu/MemesClassification/blob/master/Results/C0.png">
</p>

By iterations and loop structures, we can visit all files and collect statistics one by one and layer by layer. Subsequently, by importing and utilizing ‘Seaborn’, a python data visualization library based on ‘matplotlib’, a graph with 286,433 discrete points was drawn.Each point in the graph corresponds to a local image, and the x-axis represents the widths of images, the y-axis represents the heights of images. The origin point (0, 0) is an unreal point I made up for initializing the points set, and the point at the top-right corner is( 2000, 2000).

In addition, we have also gotten the averages of widths ,heights and the width to height ratios which are 499.864 ,461.850 and 1.748 respectively.

A glance at the graph generated reveals two noteworthy points we need to figure out:

#### 1.1.1 There are several apparent horizontal, vertical and oblique straight lines on the graph.
By means of picking out images correspond to points on those straight lines, I found all images that with the same widths or heights are complete and incomplete screen-shots made from electronic equipment in various sizes screens. 
However, most of products are manufactured under some specific industry rules and developed along the course of their own. For example, the mainstream of the widths and heights to current screens prefers 16 : 9.
As a result, screen-shots recorded by equipment will follow similar laws. We can even predict the graph that shows how much does the proportion of ever model account for in all types of screens, by analyzing all kinds of screen-shots. For now, however, these lines can be the constraints for us to do initial selections.

#### 1.1.2 Majority of points focus on a specific area.
Restricted to the media transmission rules which are made by developers and services providers for saving costs and improving efficiency, most of images will be compressed to relatively appropriate sizes before uploading, therefore it is inevitably that over 90 percent received images which have been spreading on Internet will gather around a clear resolution area like the zone shown on the graph. And the zone which has the biggest density of images is the triangle area between (0,0) and (720, 1080).

### 1.2 Graph [2]
Subsequently, I picked 3,037 genuine ‘Social Function Pictures’ by manually recognition, with the averages of widths, heights and the width to height ratios are 193.877 185.261 and 1.074 respectively, and both graphs are posted below.

#### 1.2.1 Resolutions of them largely follow a linear regression, that most of their shapes prefer converging towards squares, whereas, the heights are like to be marginally shorter than widths.
Similar to the super wide scenes presented by our eyes, as ways of recording or simulating the real world which can make us harbor the stronger sense of ‘immersion’, images contain realistic content prefer high width to height ratios, like 2:1 vision or higher. We can find that, though wide visions can make us feel calm and relaxed.
Influenced by this, the images in our life are likely to be reasonably wider.

<p align="center">
  <img width="250" height="250" src="https://github.com/SylvanLiu/MemesClassification/blob/master/Results/B0.png">
  <img width="250" height="250" src="https://github.com/SylvanLiu/MemesClassification/blob/master/Results/B1.png">
</p>

#### 1.2.1-1 However, to the SFP., why are the diversities between their widths and heights not so apparent as we found before on general images?
By observing and matching all the SFP. with final forms, not only can we find nearly every one of them actually just revolves around a head, especially a set of anthropomorphic facial features on the head, instead of the full image, and sometimes mingled with extra body movements or poses, this also suits general portrait-style images and the prototype of SFP. 
For example, the following sample is very typical for illustrating this thought, it is a standard size of wide silver screen, but the only place need to be highlighted in it is the facial expression of this actor. Hence it is quite easy to figure out where the kernel is of this pictures.

<p align="center">
  <img src="https://github.com/SylvanLiu/MemesClassification/blob/master/Results/Demo_1.png">
</p>

#### 1.2.1-2 But why all normal images we’ve received are rectangles instead of circles? 
It is not the main task for us, but I still want to try to provide a reason for this.
In our life, making things in circles needs far more material costs and far more difficult manufacturing techniques than making things with same functions but in rectangles.
Furthermore, it’s far more reasonable and easier for our ancestors to divide whole things like skins along simple lines at the first time they had found ways, because separating things into circle and circle requires relatively complex mathematical knowledge or it would be a hard time finding a valid approach to shrink and dispose the confusing waste.For most situation, things in circles would only be privilege for the people in high degrees like royal families, or be used for some specific places like wheels of vehicles. 
On the contrary, things in rectangles will be cheaper, then be well used and accepted by common people. However, the majority of things were created by ordinary people and working people. So, with a long time development of human civilization, our world actually is ‘Things in rectangles’ now, our paintings and films are shown on paper and screens in rectangles, our buildings and vehicles where we stay are cubes. It is also very feasible that all images are in rectangles inside our mobile devices which are cubes.
By way of conclusion, the best way of carrying information from the circle kernel of a picture is making an external square of it. And sometimes the kernels can be ellipses, then the figures generated will be rectangles.

...

#### Partially translated, still in progress.

## REFERENCE

[1] Yu Z . Image based Static Facial Expression Recognition with Multiple Deep Network Learning[C]// Acm on International Conference on Multimodal Interaction. ACM, 2015. [2] Krizhevsky A , Sutskever I , Hinton G . ImageNet Classification with Deep Convolutional Neural Networks[C]// NIPS. Curran Associates Inc. 2012.

[3] Cowen A S , Keltner D . Self-report captures 27 distinct categories of emotion bridged by continuous gradients[J]. Proceedings of the National Academy of Sciences, 2017:201702247.

[4] David Eberly, Perspective Mappings[J]. Geometric Tools, Redmond WA 98052 , 2011. [5] Plutchik R . A psychoevolutionary theory of emotion.[J]. Emotion Theory Research & Experience, 2000, 21(4-5):529-553.

[6] Pei S C , Lin C N . Image normalization for pattern recognition[J]. Image and Vision Computing, 1995, 13(10):711-723.

[7] Zhang S J , Cao X B , Zhang F , et al. Monocular vision-based iterative pose estimation algorithm from corresponding feature points[J]. Science in China Series F (Information Science), 2010, 53(8):1682-1696.

[8] Kazemi V , Sullivan J . One Millisecond Face Alignment with an Ensemble of Regression Trees[C] 2014 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2014.

[9] Liu N , Wang K , Jin X , et al. Visual affective classification by combining visual and text features[J]. PLoS ONE, 2017, 12(8):e0183018..

[10] Goodfellow I J, Pouget-Abadie J, Mirza M, et al. Generative adversarial nets[C] International Conference on Neural Information Processing Systems. 2014.

[11] Ahmad K . Affective Computing and Sentiment Analysis[J]. IEEE Intelligent Systems, 2016, 31(2):102-107.

[12] 刘剑聪. 社交图像美学分类与优化算法研究[D]. 2014.

[13] 赵思成. 图像情感感知的计算与应用研究[D]. 2016.

[14] 黄杰雄. 社交图像的情感和美学评价研究[D]. 2018.

[15] 刘海龙, 李宝安, 吕学强, et al. 基于深度卷积神经网络的图像检索算法研究[J]. 计算机应用研究, 2017(12):302-305.
