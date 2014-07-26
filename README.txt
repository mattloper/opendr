The OpenDR (or Open Differentiable Renderer) can be used for rendering and optimisation to image evidence.

- Rendering is achieved by constructing and initialising one of the renderers available in OpenDR.
- Optimization is achieved by minimising an objective that includes a renderer.

OpenDR is useful even without optimisation: if you just wish to animate meshes over time without performing optimisation, OpenDR may still be a good choice for rendering those meshes to images offscreen, for later compilation into a movie.

OpenDR is also designed to be concise: it should not take many lines of code to construct a scene. Although it will not replace dedicated commercial platforms (such as Maya) for some tasks, it is flexible enough for many tasks, and in some cases goes beyond what Maya can offer.

OpenDR comes with its own demos, which can be seen by typing the following::

>> import opendr
>> opendr.demo() # prints out a list of possible demos

Licensing is specified in the attached LICENSE.txt.

