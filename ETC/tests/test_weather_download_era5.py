"""Offline numerical and format checks; never request remote ERA5 data."""

import base64
import copy
import datetime as dt
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numcodecs
import numpy as np

WEATHER_SOURCE = Path(__file__).resolve().parents[2] / "01_labs/04_weather_forecasting/source_code"
_spec = importlib.util.spec_from_file_location("weather_download_under_test", WEATHER_SOURCE / "download_era5.py")
era5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(era5)


class RelativeHumidityTests(unittest.TestCase):
    def test_ice_mixed_phase_freezing_and_warm_reference_values(self):
        # Scalar reference evaluations of the ECMWF mixed-phase equations.
        # Warm cases catch the historical Earth2Studio alpha > 1 regression;
        # the mixed-phase case has alpha = 0.25 rather than a linear weight.
        temperature = np.array([240.0, 261.66, 273.16, 300.0, 310.0])
        humidity = np.array([0.0001, 0.001, 0.002, 0.01, 0.01])
        pressure = np.array([500.0, 700.0, 850.0, 1000.0, 1000.0])
        expected = np.array(
            [29.5371024049, 48.0895834385, 44.6636049595,
             45.2505826572, 25.6708295564], dtype=np.float32
        )
        # Test pressures separately so the contract does not require an array
        # pressure argument; the production channel has a scalar pressure.
        actual = np.array([
            era5.relative_humidity_percent(
                np.array([t]), np.array([q]), p
            ).item()
            for t, q, p in zip(temperature, humidity, pressure)
        ])
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)

    def test_percent_units_shape_dtype_and_upper_clipping(self):
        temperature = np.full((2, 2), 300.0)
        humidity = np.array([[0.0, 0.01], [0.1, 0.02]])
        result = era5.relative_humidity_percent(temperature, humidity, 1000.0)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, np.dtype("float32"))
        self.assertEqual(result[0, 0], 0.0)
        self.assertEqual(result[1, 0], 100.0)
        self.assertAlmostEqual(float(result[0, 1]), 45.2505826572, places=4)
        self.assertTrue(np.all((result >= 0) & (result <= 100)))

    def test_tiny_negative_raw_q_clips_only_final_rh_and_preserves_input(self):
        # Observed signed ERA5 q500 value must remain in the raw source array.
        # Diagnostics distinguish final RH clipping from pre-clipping q to 0.
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                humidity = np.array([
                    [-5.4109841585e-7, 0.0], [0.01, 0.1]
                ], dtype=dtype)
                raw_bytes = humidity.tobytes()
                temperature = np.full(humidity.shape, 300.0, dtype=dtype)
                result, diagnostics = era5.relative_humidity_percent(
                    temperature, humidity, 500.0, return_diagnostics=True
                )
                self.assertEqual(humidity.tobytes(), raw_bytes)
                self.assertEqual(result.shape, humidity.shape)
                self.assertEqual(result.dtype, np.dtype("float32"))
                self.assertEqual(result[0, 0], 0.0)
                self.assertEqual(result[0, 1], 0.0)
                self.assertEqual(result[1, 1], 100.0)
                self.assertEqual(diagnostics["raw_q_negative_count"], 1)
                self.assertEqual(diagnostics["lower_clipped_count"], 1)
                self.assertEqual(diagnostics["upper_clipped_count"], 1)
                self.assertIs(diagnostics["raw_q_modified"], False)
                self.assertLess(diagnostics["unclipped_rh_min_percent"], 0.0)
                self.assertEqual(diagnostics["raw_q_min_kgkg"], float(humidity[0, 0]))
                np.testing.assert_array_equal(
                    result,
                    era5.relative_humidity_percent(temperature, humidity, 500.0),
                )
                self.assertEqual(humidity.tobytes(), raw_bytes)

    def test_invalid_physical_inputs_are_rejected(self):
        cases = [
            (0.0, 0.01, 1000), (-1.0, 0.01, 1000),
            (np.nan, 0.01, 1000), (np.inf, 0.01, 1000),
            (300, -2.0, 1000), (300, 1.0, 1000),
            (300, np.nan, 1000), (300, np.inf, 1000),
            (300, 0.01, 0), (300, 0.01, -100),
        ]
        for temperature, humidity, pressure in cases:
            with self.subTest(t=temperature, q=humidity, p=pressure):
                with self.assertRaises(ValueError):
                    era5.relative_humidity_percent(
                        np.array([temperature]), np.array([humidity]), pressure
                    )


class ChunkDecodingTests(unittest.TestCase):
    def setUp(self):
        self.values = np.array([[[1, -2, 3], [4, 5, 6]]], dtype="<f4")
        self.metadata = {
            "zarr_format": 2, "shape": [1, 2, 3], "chunks": [1, 2, 3],
            "dtype": "<f4", "compressor": None, "filters": None,
            "order": "C", "fill_value": None,
        }

    def test_raw_chunk_decodes_exact_values_and_dimensions(self):
        actual = era5.decode_chunk(self.values.tobytes(), self.metadata)
        np.testing.assert_array_equal(actual, self.values)
        self.assertEqual(actual.shape, (1, 2, 3))

    def test_blosc_chunk_decodes_without_changing_axis_order(self):
        codec = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
        metadata = self.metadata | {"compressor": codec.get_config()}
        actual = era5.decode_chunk(codec.encode(self.values), metadata)
        np.testing.assert_array_equal(actual, self.values)

    def test_nonfinite_decoded_values_are_rejected(self):
        for value in (np.nan, np.inf):
            values = self.values.copy()
            values[0, 0, 0] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    era5.decode_chunk(values.tobytes(), self.metadata)

    def test_wrong_decoded_byte_count_is_rejected(self):
        payload = self.values.tobytes()
        for wrong in (payload[:-4], payload + b"\x00\x00\x00\x00"):
            with self.subTest(size=len(wrong)):
                with self.assertRaises(ValueError):
                    era5.decode_chunk(wrong, self.metadata)

    def test_unsupported_storage_contract_is_rejected(self):
        cases = [
            {"zarr_format": 3}, {"order": "F"}, {"dtype": "|O"},
            {"dtype": "<U4"}, {"filters": [{"id": "delta", "dtype": "<f4"}]},
        ]
        for change in cases:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    era5.decode_chunk(self.values.tobytes(), self.metadata | change)


class CanonicalFieldTests(unittest.TestCase):
    def setUp(self):
        # Storage axes are time, longitude, pressure level, latitude.
        # Distinct axis signatures expose level-selection and transpose bugs.
        self.chunk = np.array([
            [[[500, 501, 502], [1000, 1001, 1002]],
             [[510, 511, 512], [1010, 1011, 1012]]]
        ], dtype=np.float32)
        self.dimensions = ["time", "longitude", "level", "latitude"]
        self.levels = np.array([500, 1000])

    def test_exact_level_transpose_and_south_pole_crop(self):
        actual = era5.canonical_field(
            self.chunk, self.dimensions, level=1000, levels=self.levels
        )
        np.testing.assert_array_equal(actual, [[1000, 1010], [1001, 1011]])

    def test_surface_crop_can_be_disabled(self):
        surface = np.array([[[1, 2, 3]], [[11, 12, 13]]], dtype=np.float32)
        actual = era5.canonical_field(
            surface, ["longitude", "time", "latitude"], drop_south_pole=False
        )
        np.testing.assert_array_equal(actual, [[1, 11], [2, 12], [3, 13]])

    def test_missing_pressure_level_is_not_approximated(self):
        with self.assertRaises(ValueError):
            era5.canonical_field(
                self.chunk, self.dimensions, level=850, levels=self.levels
            )

    def test_ambiguous_duplicate_pressure_coordinate_is_rejected(self):
        with self.assertRaises(ValueError):
            era5.canonical_field(
                self.chunk, self.dimensions, level=500, levels=[500, 500]
            )

    def test_unknown_duplicate_or_non_singleton_time_axes_are_rejected(self):
        cases = [
            (np.ones((1, 2, 3)), ["time", "longitude", "banana"]),
            (np.ones((1, 2, 3)), ["time", "latitude", "latitude"]),
            (np.ones((2, 2, 3)), ["time", "longitude", "latitude"]),
        ]
        for values, dimensions in cases:
            with self.subTest(dimensions=dimensions, shape=values.shape):
                with self.assertRaises(ValueError):
                    era5.canonical_field(values, dimensions)

    def test_nonfinite_values_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            field = np.ones((1, 2, 3), dtype=np.float32)
            field[0, 0, 0] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    era5.canonical_field(field, ["time", "longitude", "latitude"])


class FileVerificationTests(unittest.TestCase):
    def test_size_remote_md5_and_local_sha256(self):
        # Published standard digest vectors keep the expected checks independent
        # from the implementation's digest calculations.
        metadata = {"size": "3", "md5Hash": "kAFQmDzST7DWlj99KOF/cg=="}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk"
            path.write_bytes(b"abc")
            result = era5.verify_file(path, metadata)
        self.assertEqual(result["size"], 3)
        self.assertEqual(
            result["sha256"],
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        self.assertIn(result["md5"], (
            "900150983cd24fb0d6963f7d28e17f72", metadata["md5Hash"]
        ))

    def test_size_and_same_size_content_tampering_are_rejected(self):
        metadata = {
            "size": "3",
            "md5Hash": base64.b64encode(hashlib.md5(b"abc").digest()).decode(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk"
            for wrong in (b"ab", b"abcd", b"abd"):
                path.write_bytes(wrong)
                with self.subTest(content=wrong):
                    with self.assertRaises(ValueError):
                        era5.verify_file(path, metadata)


class RequestContractTests(unittest.TestCase):
    @staticmethod
    def metadata_fixture():
        # Deliberately put time second to catch positional chunk-key guessing.
        metadata = {}
        surface = {
            "10m_u_component_of_wind": "m s**-1",
            "10m_v_component_of_wind": "m s**-1",
            "2m_temperature": "K", "surface_pressure": "Pa",
            "mean_sea_level_pressure": "Pa",
            "total_column_water_vapour": "kg m**-2",
            "100m_u_component_of_wind": "m s**-1",
            "100m_v_component_of_wind": "m s**-1",
        }
        pressure = {
            "temperature": "K", "u_component_of_wind": "m s**-1",
            "v_component_of_wind": "m s**-1", "geopotential": "m**2 s**-2",
            "specific_humidity": "kg kg**-1",
        }
        for name, units in (surface | pressure).items():
            dims = ["longitude", "time", "latitude"]
            shape, chunks = [1440, 100, 721], [1440, 1, 721]
            if name in pressure:
                dims.insert(2, "level")
                shape.insert(2, 37)
                chunks.insert(2, 37)
            metadata[name + "/.zarray"] = {
                "zarr_format": 2, "shape": shape, "chunks": chunks,
                "dtype": "<f4", "order": "C", "filters": None,
                "compressor": None,
            }
            metadata[name + "/.zattrs"] = {
                "_ARRAY_DIMENSIONS": dims, "units": units,
            }
        return metadata

    def test_exact_checkpoint_channel_order_and_truth_channel_order(self):
        self.assertEqual(era5.MODEL_VARIABLES, (
            "u10m", "v10m", "t2m", "sp", "msl", "t850", "u1000", "v1000",
            "z1000", "u850", "v850", "z850", "u500", "v500", "z500", "t500",
            "z50", "r500", "r850", "tcwv", "u100m", "v100m", "u250", "v250",
            "z250", "t250",
        ))
        self.assertEqual(era5.TRUTH_VARIABLES, ("u10m", "v10m", "msl", "t2m"))

    def test_only_initial_sources_are_inputs_and_only_four_future_surfaces(self):
        requests = era5.build_requests(self.metadata_fixture(), dt.datetime(1900, 1, 2))
        self.assertEqual(len(requests), 45)
        self.assertEqual(len({r["key"] for r in requests}), 45)
        initial = [r for r in requests if r["purpose"] == "initial"]
        truth = [r for r in requests if r["purpose"] == "truth"]
        self.assertEqual(len(initial), 13)
        self.assertEqual(len({r["variable"] for r in initial}), 13)
        self.assertEqual({r["lead_hours"] for r in initial}, {0})
        self.assertEqual(len(truth), 32)
        self.assertEqual({r["lead_hours"] for r in truth}, set(range(6, 49, 6)))
        expected_truth = {
            "10m_u_component_of_wind", "10m_v_component_of_wind",
            "mean_sea_level_pressure", "2m_temperature",
        }
        for lead in range(6, 49, 6):
            records = [r for r in truth if r["lead_hours"] == lead]
            self.assertEqual({r["variable"] for r in records}, expected_truth)
        for request in requests:
            positions = request["key"].split("/")[1].split(".")
            self.assertEqual(int(positions[1]), 24 + request["lead_hours"])
            self.assertTrue(all(int(p) == 0 for i, p in enumerate(positions) if i != 1))

    def test_requests_outside_archive_shape_are_rejected(self):
        with self.assertRaises(ValueError):
            era5.build_requests(self.metadata_fixture(), dt.datetime(1900, 1, 3, 12))

    def test_weather_units_grid_dtype_and_time_chunk_are_validated(self):
        source = "specific_humidity"
        metadata = self.metadata_fixture()
        era5.validate_weather_metadata(metadata, source)
        cases = [
            ("/.zattrs", "units", "g kg**-1"),
            ("/.zarray", "dtype", "<f8"),
            ("/.zarray", "chunks", [1440, 2, 37, 721]),
            ("/.zarray", "shape", [1440, 100, 37, 720]),
        ]
        for suffix, key, value in cases:
            altered = copy.deepcopy(metadata)
            altered[source + suffix][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    era5.validate_weather_metadata(altered, source)

    def test_time_is_naive_utc_at_an_exact_hour(self):
        expected = dt.datetime(2022, 9, 1)
        for text in ("2022-09-01T00:00:00Z", "2022-09-01T00:00:00+00:00"):
            self.assertEqual(era5.parse_time(text), expected)
        for text in (
            "2022-09-01T00:30:00Z", "2022-09-01T00:00:01Z",
            "2022-09-01T00:00:00.001Z", "2022-09-01T00:00:00+09:00",
            "not a date",
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    era5.parse_time(text)


if __name__ == "__main__":
    unittest.main()
