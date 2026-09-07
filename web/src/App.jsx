import { useEffect, useRef, useState } from 'react'
import './App.css'

function format(value, digits = 1) {
  return value == null ? '—' : value.toFixed(digits)
}

async function readResponse(response) {
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    const detail = body?.detail
    const message = Array.isArray(detail) ? detail.map((item) => item.msg).join('; ') : detail
    throw new Error(message || `Request failed (${response.status}).`)
  }
  return response.json()
}

function App() {
  const [site, setSite] = useState(null)
  const [imageReady, setImageReady] = useState(false)
  const [loadError, setLoadError] = useState('')
  const [loadAttempt, setLoadAttempt] = useState(0)
  const [start, setStart] = useState(null)
  const [goal, setGoal] = useState(null)
  const [result, setResult] = useState(null)
  const [routeError, setRouteError] = useState('')
  const [pending, setPending] = useState(false)
  const routeRequest = useRef(null)

  useEffect(() => {
    const controller = new AbortController()
    fetch('/api/site', { signal: controller.signal }).then(readResponse).then(setSite)
      .catch((error) => {
        if (error.name !== 'AbortError') setLoadError(error.message)
      })
    return () => controller.abort()
  }, [loadAttempt])

  useEffect(() => () => routeRequest.current?.abort(), [])

  function reset() {
    if (routeRequest.current) return
    setStart(null)
    setGoal(null)
    setResult(null)
    setRouteError('')
  }

  function retryLoad() {
    reset()
    setSite(null)
    setImageReady(false)
    setLoadError('')
    setLoadAttempt((attempt) => attempt + 1)
  }

  async function placePoint(event) {
    if (!imageReady || loadError || routeRequest.current) return
    const bounds = event.currentTarget.getBoundingClientRect()
    const u = (event.clientX - bounds.left) / bounds.width
    const v = (event.clientY - bounds.top) / bounds.height
    if (u < 0 || u >= 1 || v < 0 || v >= 1) return
    const [west_m, south_m, east_m, north_m] = site.bounds_m
    // Screen rows increase downwards, toward decreasing projected Y.
    // Bounds locate pixel corners; the API snaps to containing cell centers.
    const point = [west_m + u * (east_m - west_m), north_m - v * (north_m - south_m)]
    setResult(null)
    setRouteError('')
    if (!start || goal) {
      // A third click starts the next pair, including after a failed route.
      setStart(point)
      setGoal(null)
      return
    }
    setGoal(point)
    setPending(true)
    const controller = new AbortController()
    // Also guard clicks before React has rendered pending=true.
    routeRequest.current = controller
    try {
      const response = await fetch('/api/route', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start_xy_m: start, goal_xy_m: point }),
        signal: controller.signal,
      })
      const route = await readResponse(response)
      setStart(route.snapped_start_xy_m)
      setGoal(route.snapped_goal_xy_m)
      setResult(route)
    } catch (error) {
      if (error.name !== 'AbortError') setRouteError(error.message)
    } finally {
      routeRequest.current = null
      setPending(false)
    }
  }

  function imagePoint(point_xy_m) {
    const [west_m, south_m, east_m, north_m] = site.bounds_m
    // Inverse click mapping: the first sample center lands at (0.5,0.5).
    return [site.width_px * (point_xy_m[0] - west_m) / (east_m - west_m),
      site.height_px * (north_m - point_xy_m[1]) / (north_m - south_m)]
  }

  const statistics = result?.statistics
  const ready = imageReady && !loadError
  let instruction = 'Loading terrain…'
  if (loadError) instruction = 'Terrain could not be loaded.'
  else if (pending) instruction = 'Finding a route…'
  else if (ready) {
    if (!start) instruction = 'Click the map to choose a start point.'
    else if (!goal) instruction = 'Click the map to choose a goal.'
    else instruction = 'Click to start a new route, or use Reset to clear the map.'
  }

  return (
    <main>
      <h1>Lunar Traverse Planner</h1>
      <p>Shackleton rim · Site04 · 2 km × 2 km · 5 m per pixel</p>
      <p role="status" aria-live="polite">{instruction}</p>
      {loadError && <p role="alert">{loadError} <button onClick={retryLoad}>Retry</button></p>}
      {site && <>
        <div className="planner">
          <section aria-label="Site04 map">
            <svg className="map" viewBox={`0 0 ${site.width_px} ${site.height_px}`}
              onClick={placePoint} aria-label="Site04 terrain: click to place start and goal"
              aria-busy={pending || !ready}>
              <title>Site04 terrain with the route, start, and goal</title>
              <image key={loadAttempt} href={`${site.hillshade_url}?load=${loadAttempt}`}
                width={site.width_px} height={site.height_px}
                onLoad={() => setImageReady(true)}
                onError={() => setLoadError('The map image could not be loaded. Please try again.')} />
              {result?.route_xy_m && <polyline
                points={result.route_xy_m.map((point) => imagePoint(point).join(',')).join(' ')}
                fill="none" stroke="#d00000" strokeWidth="2" pointerEvents="none" />}
              {[start, goal].map((point, index) => {
                if (!point) return null
                const [x, y] = imagePoint(point)
                return <g key={index} transform={`translate(${x} ${y})`} pointerEvents="none">
                  <circle r="5" fill={index === 0 ? '#006bce' : '#ffdf00'} stroke="black" />
                  <text x={x > site.width_px - 20 ? -15 : 8} y={y < 15 ? 15 : -8}
                    fontSize="12" fill="black" stroke="white" strokeWidth="2"
                    paintOrder="stroke">{index === 0 ? 'S' : 'G'}</text>
                </g>
              })}
            </svg>
            <p>Map shading shows the shape of the terrain, not actual sunlight.</p>
            <p>Top: Y −9000 m · bottom: Y −11000 m<br />
              Left: X −6100 m · right: X −4100 m</p>
            <p>Start (S): {start ? `${format(start[0])}, ${format(start[1])} m` : 'not selected'}<br />
              Goal (G): {goal ? `${format(goal[0])}, ${format(goal[1])} m` : 'not selected'}</p>
            <button onClick={reset} disabled={pending || !ready}>Reset</button>
          </section>
          <section aria-labelledby="statistics-title" aria-busy={pending}>
            <h2 id="statistics-title">Route summary</h2>
            {routeError && <p role="alert">{routeError}</p>}
            {result?.status === 'no_route' && <p role="alert">No route was found between these points with the current settings.
              Try another pair.</p>}
            <dl className="main-statistics">
              <dt>Map distance</dt><dd>{format(statistics?.projected_length_m)} m</dd>
              <dt>Total climb</dt><dd>{format(statistics?.ascent_m)} m</dd>
              <dt>Steepest terrain</dt><dd>{format(statistics?.max_terrain_slope_deg)}°</dd>
              <dt>Average local shadow</dt><dd>{statistics
                ? format(100 * statistics.distance_weighted_mean_local_shadow_fraction, 2)
                : '—'}%</dd>
            </dl>
            {statistics && <details>
              <summary>Route details</summary>
              <dl>
                <dt>Route score</dt>
                <dd>{format(statistics.total_cost_weighted_m)} weighted meters — not energy</dd>
                <dt>Steepest step (up or down)</dt><dd>{format(statistics.max_abs_step_grade_deg)}°</dd>
                <dt>Diagonal steps beside blocked terrain</dt>
                <dd>{statistics.diagonal_blocked_side_steps}</dd>
                <dt>Of those, blocked on both sides</dt>
                <dd>{statistics.diagonal_both_blocked_steps}</dd>
              </dl>
            </details>}
            <p>This demo allows terrain slopes up to {site.slope_limit_deg}°.
              Slope and shadow weights are fixed at {site.slope_weight} and {site.shadow_weight}.
              These express planning preferences, not measured effort or a safety limit.</p>
            <p>Shadow conditions: {site.start_utc.replace('T', ' ')} to{' '}
              {site.end_utc.replace('T', ' ')} UTC. The percentage averages the saved shadow
              conditions along the route, weighted by distance. It does not estimate
              how long you would walk in shadow.</p>
          </section>
        </div>
        <p>Routes stay inside this map. The shadow calculation misses terrain farther away.
          Individual steps can be steeper than the terrain slope limit, and diagonal moves
          can cut between blocked cells. At 5 m per pixel, the map cannot show small obstacles
          or prove there is room to pass. This is a planning prototype, not a safety assessment.</p>
        <p>Terrain data: NASA GSFC PGDA, Site04. Positions use lunar south polar map
          coordinates in meters ({site.frame}).</p>
      </>}
    </main>
  )
}

export default App
