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
    theme = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
  var meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = theme === "light" ? "#f2f4f8" : "#0b0e17";
})();
