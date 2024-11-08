This is the page to show supplementary movies in the paper "[Integrating GNN and Neural ODEs for Estimating Non-Reciprocal Two-Body Interactions in Mixed-Species Collective Motion](https://openreview.net/forum?id=qwl3EiDi9r)", a conference paper in NeurIPS 2024.

[Back to the project page](https://github.com/MasahitoUWAMICHI/collectiveMotionNN)


<script>
function setVideoWidth(width) {
    const videos = document.querySelectorAll('video');
    videos.forEach(video => {
        video.width = width;
    });
}
</script>

<div>
    <label for="video-width">Select video width: </label>
    <select id="video-width" onchange="setVideoWidth(this.value)">
        <option value="200">200px</option>
        <option value="400">400px</option>
        <option value="600">600px</option>
        <option value="800">800px</option>
        <option value="1000">1000px</option>
    </select>
</div>

- ## Harmonic Interaction Model

    - ### Training Data

    <div style="display: flex; flex-wrap: wrap; gap: 20px;">
        <div style="flex: 1;">
            <details>
                <summary>Supplemental Movie S1</summary>
                <video width="600" controls>
                    <source src="Supplemental_Movie_S1.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <p><em>This code is a result of numerical simulation with a smaller strength of friction, \(\rho = 1 \times 10^{-2}\).</em></p>
            </details>
        <div style="flex: 1;">
            <details>
                <summary>Supplemental Movie S2</summary>
                <video width="600" controls>
                    <source src="Supplemental_Movie_S2.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <p><em>This code is a result of numerical simulation with a larger strength of friction, <span>\(\rho = 1 \times 10^{-1}\)</span>.</em></p>
            </details>
        </div>
        
        <script type="text/javascript" async
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
        </script>
    </div>

    - ### Estimated Dynamics

<div style="display: flex; flex-wrap: wrap; gap: 20px;">
    <div style="flex: 1;">
        <details>
            <summary>Supplemental Movie S3</summary>
            <video width="600" controls>
                <source src="Supplemental_Movie_S3.mp4" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </details>
    </div>
    <div style="flex: 1;">
        <details>
            <summary>Supplemental Movie S4</summary>
            <video width="600" controls>
                <source src="Supplemental_Movie_S4.mp4" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </details>
    </div>
    <div style="flex: 1;">
        <details>
            <summary>Supplemental Movie S5</summary>
            <video width="600" controls>
                <source src="Supplemental_Movie_S5.mp4" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </details>
    </div>
</div>

