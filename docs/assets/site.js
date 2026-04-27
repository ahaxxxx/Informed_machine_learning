const currentPage = document.body.dataset.page;
const navLinks = document.querySelectorAll(".site-nav a");
const navToggle = document.querySelector(".nav-toggle");
const siteNav = document.querySelector(".site-nav");

for (const link of navLinks) {
  const href = link.getAttribute("href") || "";
  if (
    (currentPage === "home" && href.endsWith("index.html")) ||
    (currentPage === "reading-map" && href.endsWith("reading-map.html")) ||
    (currentPage === "notes" && href.endsWith("notes.html")) ||
    (currentPage === "toys" && href.endsWith("toys.html")) ||
    (currentPage === "deploy" && href.endsWith("deploy.html"))
  ) {
    link.classList.add("is-active");
  }
}

if (navToggle && siteNav) {
  navToggle.addEventListener("click", () => {
    const expanded = navToggle.getAttribute("aria-expanded") === "true";
    navToggle.setAttribute("aria-expanded", String(!expanded));
    siteNav.classList.toggle("is-open", !expanded);
  });
}
