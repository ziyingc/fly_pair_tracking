# Fly pair tracking for Kestrel System for fly assay  with DeepLabCut

This is a collection of python module files and jupyter notebook for fly tracking method for Paper "High-Throughput Tracking of Freely Moving Drosophila Reveal Variations in Social Behaviors". The main goal is to provide highreduce number of the "NaN" pose estimation frames from DLC model inference result becuase flies were moving with very little limitation in our experiment setup. So 3 models were trained based on whether a fly shows the top, side or cross section to the camera angle. And a view integration filter was developed to combine 3 inferencing result. Videos used were recorded using [Kestrel imaging system, Ramona Optics](https://www.ramonaoptics.com/products/kestrel).
<br>
<br>
2 driver lines of fruit fly were used in this paper. This repo is an example to analyze 1 driver line (r72a10-gal4 experiments), 86 total videos from multiple recording days.

![My Example Image](images/my-image.png)

## Description
Jupyter notebooks in [DataStreamline](DataStreamLine) are the workflow of analyzing the data after experiments, 12 steps. Some functions with repeated usage in the study are in the python modules in [numphly](numphly).
* ***Prepare experiment recording video for analysis:01***<br>
This includes identifying data directory and convert video to compressed version which is portable. and add frame number to each frame for easir and accurate timestamp identification.

* ***Manually input information of each fly pair:02***<br>
This includes sex (-1, 0, 1), genotype(-1, 0, 1, 2), where -1 means the experiment is not viable for analysis (empty chamber or fly not healthy) starting frame index, behavior of interest (BOI) frame index array, and BOI type index array (0,1,2,3....). Other info can be added here for future experiments

* ***transfer orginal videos of viable experiments to a target directory:03***<br>

* ***Pre-DLC-inference Preparations:04***<br>
MCAM automatically crops out fly-fly video of 576 by 576 pixels based on the custom chamber after a recording session. However, further cropping was performed to recenter the arena in each video due to slight differences in chamber location in the camera view and to accelerate the DLC inference process. Using HoughCircles() in the opencv2 Python module, the center and inner radius of the arena were determined in pixel units. The inner radius of the arena in pixel units across all videos ranged between 225 to 235 for a physical size of 8.4 mm, so a pixel-to-physical scale ratio was calculated for each fly-fly video. During DLC inference, further cropping was performed based on the center location of the arena and extending 240 pixels in all 4 directions, which covers the entire arena inner radius.

* ***Clip BOI based on manaul score for reviewing and labeling in DeepLabCut UI interface:05***<br>
3 models of DLC tracking were prepared based on fly body orientation in the video, top view, side view and cross view.
* ***DLC implementation: training and inferencing***:06 and 07<br>
top view model and side view model are labeled using manual scored clips of BOI and cross view model were trained based on the NaN inference results from top view and side view models.
* ***Preview the tracking results from the 3 models:08***<br>
This step is an reality check, to see how many frames had viable tracking results based on different body parts.
* ***Fly Body Segment Length Measurement:09***<br>
The body segment length of an individual fly was measured using a section of video recording with fly pair pose estimation results of high confidence from either the top or side view model. The section of pose estimation result was an at least 1-second-long time window meeting 3 conditions:<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;1. All head, center, and tail points are detected from both flies at every frame in the time window.<br>
&nbsp;&nbsp;&nbsp;&nbsp;2. The head-to-head, tail-to-tail, and head-to-tail distances between the fly pair are at least 1 mm apart to guarantee clear separation of the two flies in view.<br>
&nbsp;&nbsp;&nbsp;&nbsp;3. With a 6mm distance between the center body part and the arena center as the threshold, it is decided whether the section is from the top or side view model. If both center body parts are within the 6mm threshold, then both flies are on the floor in the top view angle, and the section for body segment length measurement is from the top view model pose estimation result. If both center body parts are outside the 6 mm threshold, the side view model pose estimation is used.<br><br>
The average distance between the front part, 𝑓, and rear part, 𝑟, is defined as the length of the body segment, 𝑑(𝑓, 𝑟). The front and rear parts are head and tail for the whole body segment, head and center for the head+thorax segment, or center and tail for the abdomen segment.

* ***View-Selection Filter for Multi-View Integration:10***<br>
The view-selection filter is based on DLC pose estimations’ confidence scores and the fly’s realistic physical size. Following prioritized steps and rules, the filter fills the gap in head, center, and tail detections of the fly pair in the top view model baseline to form the multi-view coordinate array (MVCA).

* ***Basical locomotive feature measurements:11 and 12***<br>



### Dependencies

* deeplabcut



