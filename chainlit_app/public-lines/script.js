window.addEventListener("DOMContentLoaded", () => {
    const splash = document.createElement("div");
    splash.id = "splash-screen";
    const img = document.createElement("img");
    img.src = "/public/logo.png";
    img.alt = "Logo";
    img.style.width = "120px";
    splash.appendChild(img);
    document.body.appendChild(splash);

    setTimeout(() => {
        splash.classList.add("fade-out");
        setTimeout(() => {
            splash.remove();
        }, 250);
    }, 1000);
});
