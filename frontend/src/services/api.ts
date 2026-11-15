export async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...options,
    signal:
      options?.signal ||
      AbortSignal.timeout(
        path.startsWith("/research/predict") ||
          path.startsWith("/analyze-audio")
          ? 130000
          : 20000,
      ),
  });
  if (!res.ok) {
    let message = `Le service a retourné une erreur (${res.status}).`;
    try {
      const body = await res.json();
      message =
        typeof body.detail === "string"
          ? body.detail
          : "Vérifiez les champs saisis.";
    } catch {
      /* keep readable HTTP error */
    }
    throw new Error(message);
  }
  return res.json();
}
