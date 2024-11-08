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
    // Select the target section by ID
    const section = document.getElementById(sectionId);
    
    // Select the Training Data and Estimated Dynamics subsections within the section
    const subsections = Array.from(section.querySelectorAll('h3')).filter(h3 => 
        h3.textContent.includes("Training Data") || h3.textContent.includes("Estimated Dynamics")
    );
    
    // Iterate through each subsection
    subsections.forEach(subsection => {
        // Find the next sibling element which should be the container for the details
        let nextElement = subsection.nextElementSibling;
        
        // Ensure the next element is a div
        if (nextElement && nextElement.tagName.toLowerCase() === 'div') {
            // Find all open <details> elements within the div
            const openDetails = nextElement.querySelectorAll('details[open]');
            
            // Iterate through each open <details> element
            openDetails.forEach(details => {
                // Find all <video> elements within the open <details> element
                const videos = details.querySelectorAll('video');
                
                // Play each video
                videos.forEach(video => {
                    video.play();
                });
            });
        }
    });
}
</script>

- ## Harmonic Interaction Model
    <div id="harmonic-interaction-model" style="margin-top: 20px;">
        <button onclick="playAllVideos('harmonic-interaction-model')">Play Open Videos</button>
        
        - ### Training Data
            <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                <div style="flex: 1;">
                    <details>
                        <summary>Supplemental Movie S1</summary>
                        <video width="400" controls>
                            <source src="Supplemental_Movie_S1.mp4" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </details>
                </div>
                <div style="flex: 1;">
                    <details>
                        <summary>Supplemental Movie S2</summary>
                        <video width="400" controls>
                            <source src="Supplemental_Movie_S2.mp4" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </details>
                </div>
            </div>
        
        - ### Estimated Dynamics
            <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                <div style="flex: 1;">
                    <details>
                        <summary>Supplemental Movie S3</summary>
                        <video width="400" controls>
                            <source src="Supplemental_Movie_S3.mp4" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </details>
                </div>
                <div style="flex: 1;">
                    <details>
                        <summary>Supplemental Movie S4</summary>
                        <video width="400" controls>
                            <source src="Supplemental_Movie_S4.mp4" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </details>
                </div>
            </div>
    </div>






