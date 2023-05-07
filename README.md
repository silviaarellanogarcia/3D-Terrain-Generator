# 3D-Terrain-Generator

## Settings
The 3D Terrain Generator is only available locally. Therefore, to use it you have to download this repository.
```bash
$ git clone git@github.com:silviaarellanogarcia/3D-Terrain-Generator.git
```
Then, you have to install the necessary libraries to execute the tool
```bash
$ pip install -r requirements.txt 
```
The tool is now ready to use! To launch it use the following commands:
```bash
$ export FLASK_APP=temp
$ export FLASK_ENV=production
$ flask run
```
And press the link outputted by the terminal
```bash
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```
## How to generate your terrain?
Start by pressing the ```Generate Heightmap button```.
Once  the heightmap appears on the right side of the screen, decide if you want to keep it by clicking the ```Choose this one!``` button, or start 
the process again by pressing ```Regenerate heightmap```.

Once you are satisfied with your heightmap, scroll down to the Step 2 and create the textures. You can add as many as you wish. To do so, just press the 
```Add texture``` button. To get rid of a texture, just press the cross next to the texture you want to delete.

Don't forget to customize them! The texture can be chosen between several options, displayed in a drop-down list. Then, you have to choose in which range 
of textures would you like to apply it. In the example shown below, the mud is applied to the areas of the image that have a grey level between 0% and 
20%, represented as a percentage instead of the usual 0-255 range. Lastly, you can decide whether you want to apply the texture to the full image, or just 
in a specific region of the image. When you are finished adding and customizing textures, click on ```Apply!```.

Your terrain is ready to go! To check if your terrain meets your expectations, click on ```Render terrain```. Then, press ```Download the render files````
 to obtain the .obj and .mtl of your terrain. There will be two types of files downloaded. On the one hand, all the terrain as a whole will be available
 under the name ```MyTerrain```. On the other hand, if you want to assign different surface parameters to each texture, it would be more practical to use
 the files named as ```Separate_X```, that contain the mesh and texture of each part.
 
Finally, you can introduce the files into your robotics simulator!
