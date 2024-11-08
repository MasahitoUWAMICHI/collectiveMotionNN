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
        <option value="400" selected>400px</option>
        <option value="600">600px</option>
        <option value="800">800px</option>
        <option value="1000">1000px</option>
    </select>
</div>

<script type="text/javascript" async
src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


<script>
function playAllVideos(sectionId) {
    console.log('playAllVideos called with sectionId:', sectionId); // Debugging line

    // Select the target section by ID
    const section = document.getElementById(sectionId);
    if (!section) {
        console.error('Section not found:', sectionId);
        return;
    }
    
    // Select the Training Data and Estimated Dynamics subsections within the section
    const subsections = Array.from(section.querySelectorAll('[data-subsection]')).filter(subsection => 
        subsection.dataset.subsection === "Training Data" || subsection.dataset.subsection === "Estimated Dynamics"
    );
    
    // Iterate through each subsection
    subsections.forEach(subsection => {
        // Find all open <details> elements within the subsection
        const openDetails = subsection.querySelectorAll('details[open]');
        
        // Iterate through each open <details> element
        openDetails.forEach(details => {
            // Find all <video> elements within the open <details> element
            const videos = details.querySelectorAll('video');
            
            // Play each video
            videos.forEach(video => {
                video.play();
            });
        });
    });
}
</script>

<h2>Harmonic Interaction Model</h2>
<div id="harmonic-interaction-model" style="margin-top: 20px;">
    <button onclick="playAllVideos('harmonic-interaction-model')">Play Open Videos</button>
    
    <h3>Training Data</h3>
    <div data-subsection="Training Data" style="display: flex; flex-wrap: wrap; gap: 20px;">
        <div style="flex: 1;">
            <details>
                <summary>Supplemental Movie S1</summary>
                Small friction constant \(\gamma = 1\times 10^{-2} \)
                <video width="400" controls>
                    <source src="Supplemental_Movie_S1.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </details>
        </div>
        <div style="flex: 1;">
            <details>
                <summary>Supplemental Movie S2</summary>
                Large friction constant \(\gamma = 1\times 10^{-1} \).
                <video width="400" controls>
                    <source src="Supplemental_Movie_S2.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </details>
        </div>
    </div>
    
    <h3>Estimated Dynamics</h3>
    <div data-subsection="Estimated Dynamics" style="display: flex; flex-wrap: wrap; gap: 20px;">
        <div style="flex: 1;">
            <details>
                <summary>Supplemental Movie S3</summary>
                Small friction constant \(\gamma = 1\times 10^{-2} \); Many trials \(N_{tra} = 270\) in the training data.
                <video width="400" controls>
                    <source src="Supplemental_Movie_S3.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </details>
        </div>
        <div style="flex: 1;">
            <details>
                <summary>Supplemental Movie S4</summary>
                Small friction constant \(\gamma = 1\times 10^{-2} \); Few trials \(N_{tra} = 3\) in the training data.
                <video width="400" controls>
                    <source src="Supplemental_Movie_S4.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </details>
        </div>
        <div style="flex: 1;">
            <details>
                <summary>Supplemental Movie S5</summary>
                Large friction constant \(\gamma = 1\times 10^{-1} \); Few trials \(N_{tra} = 3\) in the training data.
                <video width="400" controls>
                    <source src="Supplemental_Movie_S5.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </details>
    </div>
</div>


<h2>Mixed-Species Model</h2>
<div id="mixed-species-model" style="margin-top: 20px;">
    <button onclick="playAllVideos('mixed-species-model')">Play Open Videos</button>
    
    <h3>Training Data</h3>
    <div data-subsection="Training Data" style="display: flex; flex-wrap: wrap; gap: 20px;">
        <div style="flex: 1;">
            <details>
                <summary>Supplemental Movie S6</summary>
                Parameter set (i)
                <video width="400" controls>
                    <source src="Supplemental_Movie_S6.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </details>
        </div>
        <div style="flex: 1;">
            <details>
                <summary>Supplemental Movie S7</summary>
                Parameter set (ii)
                <video width="400" controls>
                    <source src="Supplemental_Movie_S7.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </details>
        </div>
    </div>
    
    <h3>Estimated Dynamics</h3>
    <div data-subsection="Estimated Dynamics" style="display: flex; flex-wrap: wrap; gap: 20px;">
        <div style="flex: 1;">
            <details>
                <summary>Supplemental Movie S8</summary>
                Parameter set (i); Few trials \(N_{tra} = 3\) in the training data.
                <video width="400" controls>
                    <source src="Supplemental_Movie_S8.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </details>
        </div>
        <div style="flex: 1;">
            <details>
                <summary>Supplemental Movie S9</summary>
                Parameter set (ii); Few trials \(N_{tra} = 3\) in the training data.
                <video width="400" controls>
                    <source src="Supplemental_Movie_S9.mp4" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </details>
        </div>
    </div>



