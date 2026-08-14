from urllib.parse import parse_qs, urlsplit

import pytest

from metaflow.client.core import Flow
from metaflow.exception import MetaflowException
from metaflow.metadata_provider import MetadataProvider
from metaflow.plugins.metadata_providers.service import ServiceMetadataProvider


def _record(run_number, tags=None, system_tags=None):
    return {
        "flow_id": "ExampleFlow",
        "run_number": run_number,
        "ts_epoch": run_number * 1000,
        "tags": tags or [],
        "system_tags": system_tags or [],
    }


def test_service_run_iterator_follows_cursor_and_preserves_filters(monkeypatch):
    responses = [
        (
            [
                _record(3, system_tags=["user:aryan"]),
                _record(2, system_tags=["user:aryan"]),
            ],
            {"X-Next-Cursor": "next-page"},
        ),
        ([_record(1, system_tags=["user:aryan"])], {}),
    ]
    calls = []

    def fake_request(cls, monitor, path, method, **kwargs):
        calls.append((path, method, kwargs))
        return responses[len(calls) - 1]

    monkeypatch.setattr(ServiceMetadataProvider, "_request", classmethod(fake_request))

    records = list(
        ServiceMetadataProvider.iter_objects(
            "flow",
            "run",
            {"any_tags": "user:aryan"},
            None,
            "ExampleFlow",
            query_filters={"status:eq": "failed", "_tags:all": "prod"},
            page_size=2,
        )
    )

    assert [record["run_number"] for record in records] == [3, 2, 1]
    first_query = parse_qs(urlsplit(calls[0][0]).query)
    second_query = parse_qs(urlsplit(calls[1][0]).query)
    assert first_query == {
        "_limit": ["2"],
        "_tags:all": ["prod,user:aryan"],
        "status:eq": ["failed"],
    }
    assert second_query["_cursor"] == ["next-page"]
    assert second_query["status:eq"] == ["failed"]
    assert all(method == "GET" for _, method, _ in calls)
    assert all(options == {"return_headers": True} for _, _, options in calls)


def test_service_run_iterator_stops_on_repeated_cursor(monkeypatch):
    calls = []

    def fake_request(cls, monitor, path, method, **kwargs):
        calls.append(path)
        return [_record(len(calls))], {"X-Next-Cursor": "same-cursor"}

    monkeypatch.setattr(ServiceMetadataProvider, "_request", classmethod(fake_request))

    records = list(
        ServiceMetadataProvider.iter_objects(
            "flow", "run", None, None, "ExampleFlow", page_size=1
        )
    )

    assert len(records) == 2
    assert len(calls) == 2


@pytest.mark.parametrize("page_size", [0, -1, True, "10"])
def test_service_run_iterator_rejects_invalid_page_size(page_size):
    with pytest.raises((TypeError, ValueError), match="page_size"):
        list(
            ServiceMetadataProvider.iter_objects(
                "flow", "run", None, None, "ExampleFlow", page_size=page_size
            )
        )


@pytest.mark.parametrize("reserved", ["_limit", "_cursor"])
def test_service_run_iterator_owns_pagination_parameters(reserved):
    with pytest.raises(ValueError, match=reserved):
        list(
            ServiceMetadataProvider.iter_objects(
                "flow",
                "run",
                None,
                None,
                "ExampleFlow",
                query_filters={reserved: "value"},
            )
        )


def test_default_provider_iterator_preserves_existing_listing_behavior():
    class FakeProvider(MetadataProvider):
        TYPE = "fake"

        @classmethod
        def get_object(cls, obj_type, sub_type, filters, attempt, *args):
            return [_record(2), _record(1)]

    assert [
        record["run_number"]
        for record in FakeProvider.iter_objects("flow", "run", None, None, "Flow")
    ] == [2, 1]
    with pytest.raises(MetaflowException, match="not supported"):
        list(
            FakeProvider.iter_objects(
                "flow",
                "run",
                None,
                None,
                "Flow",
                query_filters={"status:eq": "failed"},
            )
        )


def test_service_request_can_return_success_headers(monkeypatch):
    class FakeResponse:
        status_code = 200
        headers = {"X-Next-Cursor": "cursor"}

        @staticmethod
        def json():
            return [_record(1)]

    class FakeSession:
        @staticmethod
        def get(url, headers):
            return FakeResponse()

    monkeypatch.setattr(ServiceMetadataProvider, "_INFO", "http://metadata")
    monkeypatch.setattr(ServiceMetadataProvider, "_session", FakeSession())

    body, headers = ServiceMetadataProvider._request(
        None, "/flows/ExampleFlow/runs", "GET", return_headers=True
    )

    assert body == [_record(1)]
    assert headers["X-Next-Cursor"] == "cursor"


def test_flow_runs_forwards_filters_tags_and_bounds_results(monkeypatch):
    flow = object.__new__(Flow)
    flow._namespace_check = True
    flow._current_namespace = "user:aryan"
    captured = {}

    def fake_iter_children(self, query_filters=None, page_size=None, required_tags=()):
        captured.update(
            query_filters=query_filters,
            page_size=page_size,
            required_tags=required_tags,
        )
        yield from range(4)

    monkeypatch.setattr(Flow, "_iter_children", fake_iter_children)

    runs = list(
        flow.runs(
            "prod",
            filters={"status:eq": "failed"},
            page_size=2,
            max_runs=2,
        )
    )

    assert runs == [0, 1]
    assert captured == {
        "query_filters": {"status:eq": "failed", "_tags:all": "prod"},
        "page_size": 2,
        "required_tags": ("prod",),
    }


def test_flow_runs_zero_limit_avoids_starting_iterator(monkeypatch):
    flow = object.__new__(Flow)

    def unexpected_iterator(*args, **kwargs):
        raise AssertionError("iterator should not be started")

    monkeypatch.setattr(Flow, "_iter_children", unexpected_iterator)

    assert list(flow.runs(max_runs=0)) == []
