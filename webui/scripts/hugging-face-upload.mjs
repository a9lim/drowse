export function uploadCommitRevision(stdout, repository) {
  let result;
  try {
    result = JSON.parse(stdout);
  } catch (error) {
    throw new Error("Hugging Face upload did not return valid JSON", { cause: error });
  }
  if (
    !result || typeof result !== "object" || Array.isArray(result) ||
    Object.keys(result).length !== 1 || typeof result.url !== "string"
  ) {
    throw new Error("Hugging Face upload did not return an exact commit URL");
  }
  let url;
  try {
    url = new URL(result.url);
  } catch (error) {
    throw new Error("Hugging Face upload returned an invalid commit URL", { cause: error });
  }
  const [owner, name, ...extraRepositoryParts] = repository.split("/");
  const parts = url.pathname.split("/").filter(Boolean);
  if (
    !owner || !name || extraRepositoryParts.length !== 0 ||
    url.origin !== "https://huggingface.co" || url.username || url.password ||
    url.search || url.hash || parts.length !== 4 || parts[0] !== owner ||
    parts[1] !== name || parts[2] !== "commit" || !/^[0-9a-f]{40}$/.test(parts[3])
  ) {
    throw new Error("Hugging Face upload commit URL does not match the target repository");
  }
  return parts[3];
}
