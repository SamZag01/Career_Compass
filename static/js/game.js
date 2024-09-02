// Global variables

const path = [];
let frequency;
const milestones = [];
let player = {
    x: 0,
    y: 0,
    radius: 10,
    speed: 5, // Adjust speed for smoother movement
    targetIndex: 0,
    moving: false
};

// Get the canvas and context
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const margin=50;
const playerImages = [
    'static/img/player/player1.png',
    'static/img/player/player2.png',
    'static/img/player/player3.png'
];
let currentFrame = 0;
const totalFrames = playerImages.length;
let playerImagesLoaded = [];

// Array of colors to cycle through
const colors = ['#c4afcf', '#e7dfec', '#dbcfe2', '#cfbfd9','#f3eff5'];

// Function to change the background color
function changeBackgroundColor() {
    // Get the current color index
    const currentColor = canvas.style.backgroundColor;
    const currentIndex = colors.indexOf(currentColor);

    // Calculate the next color index
    const nextIndex = (currentIndex + 1) % colors.length;

    // Set the next color
    canvas.style.backgroundColor = colors[nextIndex];
}
// The drawZigzagPath and drawFlags functions can remain unchanged

//-------------------------------------------------
function resizeCanvas() {
    const container = canvas.parentElement;
    canvas.width = container.clientWidth - margin;
    canvas.height = container.clientHeight;
    changeBackgroundColor();
}

function initializePath() {
const numPoints=12
const amplitude = 50; // Amplitude of the zigzag
const margin=50;

    path.length = 0; // Clear previous path
    frequency = canvas.width / (numPoints +1); // Frequency of the zigzag
    for (let i = 0; i < numPoints; i++) {
        let x = margin+i * (frequency-1);
        let y = canvas.height / 2 + Math.sin(i * Math.PI / 2) * amplitude;
        path.push({ x, y });
    }
    milestones.length = 0; // Clear previous milestones
    milestones.push(...path); // Initialize milestones

    // Initialize player at the start of the path
    player.x = path[0].x;
    player.y = path[0].y;
    player.targetIndex = 0;
    player.moving = false;
}

function drawZigzagPath() {
    if (path.length === 0) return; // No path to draw
    ctx.strokeStyle = 'yellow';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(path[0].x, path[0].y);
    path.forEach(point => ctx.lineTo(point.x, point.y));
    ctx.stroke();
}

function drawFlags() {
    if (milestones.length === 0) return; // No milestones to draw
    ctx.fillStyle = 'red';
    ctx.strokeStyle = 'black';
    milestones.forEach(point => {
        ctx.beginPath();
        ctx.moveTo(point.x, point.y);
        ctx.lineTo(point.x, point.y - 20); // Flag pole
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(point.x, point.y - 20); // Flag
        ctx.lineTo(point.x + 10, point.y - 15);
        ctx.lineTo(point.x, point.y - 10);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
    });
}
//-------------------------------------------------------------------------------------------
//function drawPlayer() {
////take a png image for player and assign it.
//    ctx.fillStyle = 'blue';
//    ctx.beginPath();
//    ctx.arc(player.x, player.y, player.radius, 0, Math.PI * 2);
//    ctx.fill();
//}
function preloadImages() {
    playerImages.forEach((src, index) => {
        const img = new Image();
        img.src = src;
        img.onload = () => {
            playerImagesLoaded[index] = img;
            if (playerImagesLoaded.length === totalFrames) {
                // All images loaded
                startAnimation(); // Start the animation
            }
        };
    });
}
function startAnimation() {
    player.moving = true; // Set the player to moving
    requestAnimationFrame(update); // Start the animation loop
}
//function drawPlayer() {
//    if (playerImagesLoaded[currentFrame]) {
//    const playerSizeMultiplier=2;
//// Calculate the new dimensions
//        const playerWidth = player.radius * 2 * playerSizeMultiplier;
//        const playerHeight = player.radius * 2 * playerSizeMultiplier;
//        ctx.drawImage(playerImagesLoaded[currentFrame], player.x - player.radius, player.y - player.radius, player.radius * 2, player.radius * 2);
//        // Update the current frame to cycle through the images
//        currentFrame = (currentFrame + 1) % totalFrames;
//    }
//}
function drawPlayer() {
    if (playerImagesLoaded[currentFrame]) {
        const playerSizeMultiplier = 2.5;

        // Calculate the new dimensions
        const playerWidth = player.radius * 2 * playerSizeMultiplier;
        const playerHeight = player.radius * 2 * playerSizeMultiplier;

        // Draw the image with the new dimensions
        ctx.drawImage(
            playerImagesLoaded[currentFrame],
            player.x - playerWidth / 2, // Adjust the x position to center the larger image
            player.y - playerHeight / 2, // Adjust the y position to center the larger image
            playerWidth, // Use the new width
            playerHeight // Use the new height
        );

        // Update the current frame to cycle through the images
        currentFrame = (currentFrame + 1) % totalFrames;
    }
}
preloadImages();

function movePlayer() {
    if (player.targetIndex < milestones.length) {
        const target = milestones[player.targetIndex];
        const dx = target.x - player.x;
        const dy = target.y - player.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance > player.speed) {
            const moveX = (dx / distance) * player.speed;
            const moveY = (dy / distance) * player.speed;
            player.x += moveX;
            player.y += moveY;
            player.moving = true;
        } else {
            player.x = target.x;
            player.y = target.y;
//            player.targetIndex++;
            if (player.targetIndex >= milestones.length) {
                player.targetIndex = 0;
                player.x = milestones[player.targetIndex].x;
                player.y = milestones[player.targetIndex].y;

            }
            player.moving = false;
        }
    }
}

//function showGif() {
//alert('showGif');
//const pic=['walk1.png','walk2.png', 'walk3.png']
//    ctx.drawImage(pic, canvas.width / 2 - pic.width / 2, canvas.height / 2 - pic.height / 2);
//    setTimeout(() => {
////        player.moving = false;
//        requestAnimationFrame(update);
//    }, 2000); // Adjust the delay time (2000ms = 2 seconds)
//}

function update() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawZigzagPath();
    drawFlags();
    drawPlayer();
    if (player.moving) {
        movePlayer();
        requestAnimationFrame(update);
    }
}

document.querySelectorAll('.controlButton').forEach(button => {
    button.addEventListener('click', () => {
        if (!player.moving) {
            player.targetIndex++;
            if (player.targetIndex >= milestones.length) {
                player.targetIndex = 0;
                player.x = milestones[player.targetIndex].x;
                player.y = milestones[player.targetIndex].y;
                changeBackgroundColor();
                requestAnimationFrame(update);

            }
            player.moving = true;
            requestAnimationFrame(update);
        }
    });
});
window.addEventListener('load', function() {
    window.scrollTo(0, 0);});

window.addEventListener('load', () => {
    resizeCanvas();
    initializePath();
    update(); // Start the animation loop
});

document.getElementById('quiz-main').addEventListener('click', function(event) {
  event.preventDefault(); // Prevent default anchor behavior
    resizeCanvas();
    initializePath();
    player.moving = true; // Set the player to moving
    requestAnimationFrame(update); // Start the animation loop
    movePlayer(); // Start the animation

});
