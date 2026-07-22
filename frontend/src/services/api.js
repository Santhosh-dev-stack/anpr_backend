import axios from 'axios'

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
})

export async function getCamera(cameraId) {
  const { data } = await client.get(`/api/cameras/${cameraId}`)
  return data
}

// Triggers the backend's video-reading/detection loop to actually begin —
// until this is called, the backend has loaded its models and is idle,
// not processing any frames regardless of the process having been launched.
export async function startCamera(cameraId) {
  const { data } = await client.post(`/api/cameras/${cameraId}/start`)
  return data
}

// detectionsUrl is already absolute (returned by getCamera) — axios ignores
// baseURL when given an absolute url, so this just uses the shared client.
export async function getDetections(detectionsUrl, segment) {
  const { data } = await client.get(detectionsUrl, { params: { segment } })
  return data
}

// OCR now runs on a background worker on the backend and can resolve well
// after the segment that triggered it was already fetched, so plate results
// are polled separately instead of riding along with per-segment detections.
export async function getPlateResults(detectionsUrl) {
  const { data } = await client.get(`${detectionsUrl}/plates`)
  return data
}
