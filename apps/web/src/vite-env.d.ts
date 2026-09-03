/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Base URL for the prediction API. Defaults to the relative `/api/v1`, which
   * the Vite dev server proxies to the FastAPI application.
   */
  readonly VITE_API_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
