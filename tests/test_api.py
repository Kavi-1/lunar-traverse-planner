"""HTTP checks against downloaded Site04 data; no generated terrain."""

import json
import shutil
from io import BytesIO

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from PIL import Image

from api import main
from core.clones import GOAL_XY_M, START_XY_M, terrain_window_m


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as http:
        yield http


def post_route(client, start_xy_m=START_XY_M, goal_xy_m=GOAL_XY_M):
    return client.post("/api/route", json={
        "start_xy_m": start_xy_m, "goal_xy_m": goal_xy_m,
    })


def test_site_registration_and_png(client):
    response = client.get("/api/site")
    assert response.status_code == 200
    metadata = response.json()
    assert metadata["bounds_m"] == [-6100, -11000, -4100, -9000]
    assert (metadata["width_px"], metadata["height_px"]) == (400, 400)
    assert (metadata["pixel_dx_m"], metadata["pixel_dy_m"]) == (5, -5)
    assert metadata["frame"] == "MOON_ME_DE421"
    assert metadata["start_utc"] == "2026-09-01T09:00:00"
    assert metadata["slope_limit_deg"] == 20
    assert metadata["slope_weight"] == metadata["shadow_weight"] == 2
    png = client.get(metadata["hillshade_url"])
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"
    pixels = np.asarray(Image.open(BytesIO(png.content)))
    assert pixels.shape[:2] == (400, 400)
    np.testing.assert_array_equal(pixels[:, :, 0], pixels[:, :, 1])
    assert pixels[:, :, 0].min() < pixels[:, :, 0].max()
    assert png.content == client.get(metadata["hillshade_url"]).content


def test_window_matches_existing_analysis(client):
    site = main.app.state.site
    heights_m, slopes_deg, transform_m = terrain_window_m(
        main.DATA_DIR / "Site04_final_adj_5mpp_surf.tif")
    np.testing.assert_array_equal(site["elevation_m"], heights_m)
    np.testing.assert_array_equal(site["slope_deg"], slopes_deg)
    assert site["transform_m"] == transform_m
    assert np.isfinite(site["slope_deg"][[0, -1], :]).all()
    assert np.isfinite(site["slope_deg"][:, [0, -1]]).all()


def test_nominal_route(client):
    response = post_route(client)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "route_rc" not in body
    assert body["route_xy_m"][0] == list(START_XY_M)
    assert body["route_xy_m"][-1] == list(GOAL_XY_M)
    stats = body["statistics"]
    assert stats["projected_length_m"] == pytest.approx(1858.4419177103414, abs=1e-8)
    assert stats["total_cost_weighted_m"] == pytest.approx(1947.5175345932507, abs=1e-8)
    assert stats["ascent_m"] == pytest.approx(62.5538330078125)
    assert stats["max_terrain_slope_deg"] == pytest.approx(19.538982014700526)
    assert stats["distance_weighted_mean_local_shadow_fraction"] == pytest.approx(
        0.002958036734429219)
    assert stats["max_abs_step_grade_deg"] > 20
    assert stats["diagonal_blocked_side_steps"] == 1
    assert stats["diagonal_both_blocked_steps"] == 0
    assert "NaN" not in response.text and "Infinity" not in response.text


def test_pixel_centers_and_one_cell(client):
    # Upper left pixel edges are (-6100,-9000); its center is half a 5 m
    # pixel inward: (-6097.5,-9002.5), which draws at SVG (0.5,0.5).
    response = post_route(client, (-6100, -9000), (-6096, -9004))
    assert response.status_code == 200
    body = response.json()
    assert body["route_xy_m"] == [[-6097.5, -9002.5]]
    assert body["start_snap_distance_m"] == pytest.approx(np.sqrt(12.5))
    assert body["goal_snap_distance_m"] == pytest.approx(np.sqrt(4.5))
    assert body["statistics"]["projected_length_m"] == 0
    assert body["statistics"]["ascent_m"] == 0
    assert body["statistics"]["total_cost_weighted_m"] == 0
    assert body["statistics"]["max_abs_step_grade_deg"] is None
    assert body["statistics"]["distance_weighted_mean_local_shadow_fraction"] == (
        main.app.state.site["shadow_fraction"][0, 0])


@pytest.mark.parametrize("point_xy_m", [
    (-6100.1, -10000), (-4100, -10000), (-5000, -8999.9), (-5000, -11000),
])
def test_outside_endpoints(client, point_xy_m):
    response = post_route(client, point_xy_m)
    assert response.status_code == 422
    assert response.json()["detail"] == "start is outside the raster"


@pytest.mark.parametrize("endpoint", ["start_xy_m", "goal_xy_m"])
def test_blocked_endpoints(client, endpoint):
    # Actual cell (0,71) exceeds the selected cutoff; no terrain is modified.
    body = {"start_xy_m": START_XY_M, "goal_xy_m": GOAL_XY_M}
    body[endpoint] = (-5742.5, -9002.5)
    response = client.post("/api/route", json=body)
    assert response.status_code == 422
    assert response.json()["detail"] == f"{endpoint.split('_')[0]} is blocked"


def test_naturally_disconnected_goal(client):
    # Cell (74,349) lies in a two-cell eight-connected island in the actual
    # 20-degree cost mask, separate from the nominal start's component.
    response = post_route(client, goal_xy_m=(-4352.5, -9372.5))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_route"
    assert body["route_xy_m"] is None
    assert body["statistics"] is None
    assert body["snapped_goal_xy_m"] == [-4352.5, -9372.5]


@pytest.mark.parametrize("start", [[], [1], [1, 2, 3], ["-5697.5", -10002.5], [True, 1]])
def test_malformed_coordinates(client, start):
    assert post_route(client, start).status_code == 422


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_coordinates_produce_json_error(client, literal):
    body = '{"start_xy_m":[' + literal + ',0],"goal_xy_m":[0,0]}'
    response = client.post("/api/route", content=body,
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "finite_number"


def test_settings_cannot_be_overridden(client):
    assert client.post("/api/route", json={
        "start_xy_m": START_XY_M, "goal_xy_m": GOAL_XY_M, "slope_limit_deg": 25,
    }).status_code == 422


def test_requests_reuse_prepared_data(client, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Request must not read files or rebuild terrain/costs")

    for name in ("prepare_site", "input_hash", "compute_slope_deg", "build_cost_surface"):
        monkeypatch.setattr(main, name, unexpected)
    monkeypatch.setattr(rasterio, "open", unexpected)
    assert client.get("/api/site").status_code == 200
    assert client.get("/api/hillshade.png").status_code == 200
    assert post_route(client).status_code == 200
    assert post_route(client, GOAL_XY_M, START_XY_M).status_code == 200
    assert not main.app.state.site["cost_weights"].flags.writeable


@pytest.fixture
def copied_inputs(tmp_path):
    # Symlink the unchanged real DEM; copy only small derived metadata/raster
    # files before corrupting those copies to exercise startup rejection.
    for name in ("Site04_final_adj_5mpp_surf.tif", "Site04_final_adj_5mpp_surf.tif.sha256"):
        (tmp_path / name).symlink_to(main.DATA_DIR / name)
    for name in ("site04_illumination_report.json", "site04_mean_local_shadow.tif"):
        shutil.copyfile(main.DATA_DIR / name, tmp_path / name)
    return tmp_path


def test_missing_inputs_stop_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="README"), TestClient(main.app):
        pass


@pytest.mark.parametrize("field,value", [
    ("dem_sha256", "wrong source"), ("start_utc", "2026-09-06T00:00:00"),
    ("output_shape", [399, 400]), ("mean_shadow_fraction", -1),
])
def test_reject_mismatched_report(copied_inputs, field, value):
    path = copied_inputs / "site04_illumination_report.json"
    report = json.loads(path.read_text())
    report[field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Illumination|illumination"):
        main.prepare_site(copied_inputs)


def test_reject_shadow_registration(copied_inputs):
    with rasterio.open(copied_inputs / "site04_mean_local_shadow.tif", "r+") as raster:
        raster.transform = raster.transform * rasterio.Affine.translation(1, 0)
    with pytest.raises(ValueError, match="illumination metadata"):
        main.prepare_site(copied_inputs)


def test_reject_shadow_sampling(copied_inputs):
    with rasterio.open(copied_inputs / "site04_mean_local_shadow.tif", "r+") as raster:
        raster.update_tags(time_step_seconds="150")
    with pytest.raises(ValueError, match="sampling metadata"):
        main.prepare_site(copied_inputs)


def test_reject_dem_hash(copied_inputs):
    sidecar = copied_inputs / "Site04_final_adj_5mpp_surf.tif.sha256"
    sidecar.unlink()  # Remove only the temporary symlink, not the source sidecar.
    sidecar.write_text("incorrect-hash")
    with pytest.raises(ValueError, match="Hash mismatch"):
        main.prepare_site(copied_inputs)
