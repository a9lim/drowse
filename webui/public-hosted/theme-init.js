(function () {
  if (document.activeViewTransition) document.activeViewTransition.ready.catch(function () {});
  function handlePageTransition(event) {
    if (event.viewTransition) event.viewTransition.ready.catch(function () {});
  }
  window.addEventListener("pageswap", handlePageTransition);
  window.addEventListener("pagereveal", handlePageTransition);
  var theme = null;
  try {
    theme = window.localStorage.getItem("drowse.theme");
    if (theme === null) {
      theme = window.localStorage.getItem("polythetic.theme") || window.localStorage.getItem("saklas.theme");
      if (theme !== null) {
        window.localStorage.setItem("drowse.theme", theme);
      }
    }
  } catch (_) {}
  if (theme !== "light" && theme !== "dark") {
    // No saved choice: follow the system appearance. theme.ts keeps following
    // it until someone picks a theme with the toggle.
    theme = "dark";
    try {
      if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) theme = "light";
    } catch (_) {}
  }
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
  var metas = document.querySelectorAll('meta[name="theme-color"]');
  for (var metaIndex = 0; metaIndex < metas.length; metaIndex++) {
    metas[metaIndex].content = theme === "light" ? "#f2f4f8" : "#0b0e17";
  }
  // Prerendered pages are rendered without knowing the theme. Controls that
  // carry a label for each theme get the right one before the app mounts.
  if (typeof document.addEventListener === "function") {
    document.addEventListener("DOMContentLoaded", function () {
      var controls = document.querySelectorAll("[data-theme-label-light][data-theme-label-dark]");
      for (var index = 0; index < controls.length; index++) {
        controls[index].setAttribute("aria-label", controls[index].getAttribute("data-theme-label-" + theme));
      }
    });
  }
})();
