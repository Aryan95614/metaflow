from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest

from metaflow.client.core import Flow
from metaflow.exception import MetaflowException, MetaflowInternalError
from metaflow.plugins.metadata_providers.local import LocalMetadataProvider
from metaflow.plugins.metadata_providers.service import (
    ServiceException,
    ServiceMetadataProvider,
)

PAGINATING_SERVICE_VERSION = (
    ServiceMetadataProvider._MIN_SERVICE_VERSION_WITH_CURSOR_PAGINATION
)


def _record(run_number, tags=None, system_tags=None):
    return {
        "flow_id": "ExampleFlow",
        "run_number": run_number,
        "ts_epoch": run_number * 1000,
        "tags": tags or [],
        "system_tags": system_tags or [],
    }


@pytest.fixture(autouse=True)
def reset_service_capability_cache():
    ServiceMetadataProvider._supports_cursor_pagination = None
    yield
    ServiceMetadataProvider._supports_cursor_pagination = None


def _paginating_version(cls, monitor):
    return PAGINATING_SERVICE_VERSION


def _legacy_version(cls, monitor):
    return "2.4.0"


def test_service_run_iterator_follows_cursor_and_preserves_filters(monkeypatch):
    monkeypatch.setattr(
        ServiceMetadataProvider, "_version", classmethod(_paginating_version)
    )
    responses = [
        (
            [
                _record(3, system_tags=["user:aryan"]),
                _record(2, system_tags=["user:aryan"]),
            ],
            {"X-Next-Cursor": "next-page", "X-Limit": "2"},
        ),
        ([_record(1, system_tags=["user:aryan"])], {"X-Limit": "2"}),
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


def test_service_iterator_paginates_all_collection_types(monkeypatch):
    monkeypatch.setattr(
        ServiceMetadataProvider, "_version", classmethod(_paginating_version)
    )
    calls = []

    def fake_request(cls, monitor, path, method, **kwargs):
        calls.append(path)
        return [_record(1)], {"X-Limit": "1"}

    monkeypatch.setattr(ServiceMetadataProvider, "_request", classmethod(fake_request))

    list(
        ServiceMetadataProvider.iter_objects(
            "run", "step", None, None, "ExampleFlow", "12", page_size=1
        )
    )

    assert calls[0].startswith("/flows/ExampleFlow/runs/12/steps?")
    assert parse_qs(urlsplit(calls[0]).query) == {"_limit": ["1"]}


def test_service_run_iterator_stops_on_repeated_cursor(monkeypatch):
    monkeypatch.setattr(
        ServiceMetadataProvider, "_version", classmethod(_paginating_version)
    )
    calls = []

    def fake_request(cls, monitor, path, method, **kwargs):
        calls.append(path)
        return [_record(len(calls))], {"X-Next-Cursor": "same-cursor", "X-Limit": "1"}

    monkeypatch.setattr(ServiceMetadataProvider, "_request", classmethod(fake_request))

    records = list(
        ServiceMetadataProvider.iter_objects(
            "flow", "run", None, None, "ExampleFlow", page_size=1
        )
    )

    assert len(records) == 2
    assert len(calls) == 2


def test_old_service_uses_legacy_listing_without_query_params(monkeypatch):
    monkeypatch.setattr(
        ServiceMetadataProvider, "_version", classmethod(_legacy_version)
    )
    calls = []

    def fake_request(cls, monitor, path, method, **kwargs):
        calls.append(path)
        return [_record(2), _record(1)], {}

    monkeypatch.setattr(ServiceMetadataProvider, "_request", classmethod(fake_request))

    records = list(
        ServiceMetadataProvider.iter_objects("flow", "run", None, None, "ExampleFlow")
    )

    assert [record["run_number"] for record in records] == [2, 1]
    assert calls == ["/flows/ExampleFlow/runs"]
    assert "?" not in calls[0]


def test_old_service_rejects_server_filters_without_listing(monkeypatch):
    monkeypatch.setattr(
        ServiceMetadataProvider, "_version", classmethod(_legacy_version)
    )

    def unexpected_request(*args, **kwargs):
        raise AssertionError(
            "legacy services must not receive filtered listing requests"
        )

    monkeypatch.setattr(
        ServiceMetadataProvider, "_request", classmethod(unexpected_request)
    )

    with pytest.raises(ServiceException, match="Filtering requires"):
        list(
            ServiceMetadataProvider.iter_objects(
                "flow",
                "run",
                None,
                None,
                "ExampleFlow",
                query_filters={"status:eq": "failed"},
            )
        )


def test_new_service_without_limit_header_does_not_yield_unfiltered_records(
    monkeypatch,
):
    monkeypatch.setattr(
        ServiceMetadataProvider, "_version", classmethod(_paginating_version)
    )
    yielded = []

    def fake_request(cls, monitor, path, method, **kwargs):
        return [_record(1)], {}

    monkeypatch.setattr(ServiceMetadataProvider, "_request", classmethod(fake_request))

    with pytest.raises(ServiceException, match="Filtering requires"):
        for record in ServiceMetadataProvider.iter_objects(
            "flow",
            "run",
            None,
            None,
            "ExampleFlow",
            query_filters={"status:eq": "failed"},
        ):
            yielded.append(record)

    assert yielded == []


def test_new_service_without_limit_header_falls_back_to_legacy_listing(monkeypatch):
    monkeypatch.setattr(
        ServiceMetadataProvider, "_version", classmethod(_paginating_version)
    )
    calls = []

    def fake_request(cls, monitor, path, method, **kwargs):
        calls.append((path, kwargs))
        if "return_headers" in kwargs:
            return [_record(9)], {}
        return [_record(2), _record(1)], True

    monkeypatch.setattr(ServiceMetadataProvider, "_request", classmethod(fake_request))

    records = list(
        ServiceMetadataProvider.iter_objects("flow", "run", None, None, "ExampleFlow")
    )

    assert [record["run_number"] for record in records] == [2, 1]
    assert calls[0][1] == {"return_headers": True}
    assert "?" in calls[0][0]
    assert calls[1][0] == "/flows/ExampleFlow/runs"


@pytest.mark.parametrize("page_size", [0, -1, True, "10"])
def test_service_run_iterator_rejects_invalid_page_size(page_size, monkeypatch):
    monkeypatch.setattr(
        ServiceMetadataProvider, "_version", classmethod(_paginating_version)
    )
    with pytest.raises((TypeError, ValueError), match="page_size"):
        list(
            ServiceMetadataProvider.iter_objects(
                "flow", "run", None, None, "ExampleFlow", page_size=page_size
            )
        )


@pytest.mark.parametrize("reserved", ["_limit", "_cursor"])
def test_service_run_iterator_owns_pagination_parameters(reserved, monkeypatch):
    monkeypatch.setattr(
        ServiceMetadataProvider, "_version", classmethod(_paginating_version)
    )
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


def test_default_provider_iterator_sorts_newest_first_and_rejects_server_filters(
    monkeypatch,
):
    monkeypatch.setattr(
        LocalMetadataProvider,
        "get_object",
        classmethod(lambda cls, *args, **kwargs: [_record(1), _record(2)]),
    )

    assert [
        record["run_number"]
        for record in LocalMetadataProvider.iter_objects(
            "flow", "run", None, None, "Flow"
        )
    ] == [2, 1]
    with pytest.raises(MetaflowException, match="not supported"):
        list(
            LocalMetadataProvider.iter_objects(
                "flow",
                "run",
                None,
                None,
                "Flow",
                query_filters={"status:eq": "failed"},
            )
        )


def test_service_request_can_return_success_headers(monkeypatch):
    response = Mock()
    response.status_code = 200
    response.headers = {"X-Next-Cursor": "cursor"}
    response.json.return_value = [_record(1)]

    session = Mock()
    session.get.return_value = response

    monkeypatch.setattr(ServiceMetadataProvider, "_INFO", "http://metadata")
    monkeypatch.setattr(ServiceMetadataProvider, "_session", session)

    body, headers = ServiceMetadataProvider._request(
        None, "/flows/ExampleFlow/runs", "GET", return_headers=True
    )

    assert body == [_record(1)]
    assert headers["X-Next-Cursor"] == "cursor"


def test_flow_runs_forwards_filters_and_bounds_results():
    captured = {}

    def fake_iter_children(query_filters=None, page_size=None, required_tags=()):
        captured.update(
            query_filters=query_filters,
            page_size=page_size,
            required_tags=required_tags,
        )
        yield from range(4)

    flow = Mock()
    flow._iter_children = fake_iter_children

    runs = list(
        Flow.runs.__get__(flow, Flow)(
            "prod",
            filters={"status:eq": "failed"},
            page_size=2,
            max_runs=2,
        )
    )

    assert runs == [0, 1]
    assert captured == {
        "query_filters": {"status:eq": "failed"},
        "page_size": 2,
        "required_tags": ("prod",),
    }


def test_flow_runs_tag_and_max_runs_work_without_server_filters():
    captured = {}

    def fake_iter_children(query_filters=None, page_size=None, required_tags=()):
        captured.update(
            query_filters=query_filters,
            page_size=page_size,
            required_tags=required_tags,
        )
        yield from ("newest", "older")

    flow = Mock()
    flow._iter_children = fake_iter_children

    assert list(Flow.runs.__get__(flow, Flow)("prod", max_runs=1)) == ["newest"]
    assert captured == {
        "query_filters": None,
        "page_size": None,
        "required_tags": ("prod",),
    }


def test_flow_runs_max_runs_returns_newest_from_oldest_first_provider(monkeypatch):
    monkeypatch.setattr(
        LocalMetadataProvider,
        "get_object",
        classmethod(lambda cls, *args, **kwargs: [_record(1), _record(2)]),
    )

    records = list(
        LocalMetadataProvider.iter_objects("flow", "run", None, None, "Flow")
    )
    assert [record["run_number"] for record in records[:1]] == [2]


def test_flow_runs_zero_limit_avoids_starting_iterator():
    flow = Mock()
    flow._iter_children.side_effect = AssertionError("iterator should not be started")

    assert list(Flow.runs.__get__(flow, Flow)(max_runs=0)) == []


def test_get_object_internal_returns_none_not_empty_on_404(monkeypatch):
    """A missing (404) collection must return None like the legacy path, not [].

    The paginated iterator swallows a first-page 404, so without care
    list(...) would yield [] and mask "not found" as "empty".
    """
    monkeypatch.setattr(
        ServiceMetadataProvider, "_version", classmethod(_paginating_version)
    )

    def fake_request(cls, monitor, path, method, **kwargs):
        raise ServiceException("collection not found", http_code=404)

    monkeypatch.setattr(ServiceMetadataProvider, "_request", classmethod(fake_request))

    result = ServiceMetadataProvider._get_object_internal(
        "flow", 1, "run", 2, None, None, "ExampleFlow"
    )
    assert result is None


def test_iter_objects_yields_empty_on_404(monkeypatch):
    """Streaming a missing collection yields nothing (no 404 leaking to callers)."""
    monkeypatch.setattr(
        ServiceMetadataProvider, "_version", classmethod(_paginating_version)
    )

    def fake_request(cls, monitor, path, method, **kwargs):
        raise ServiceException("collection not found", http_code=404)

    monkeypatch.setattr(ServiceMetadataProvider, "_request", classmethod(fake_request))

    assert (
        list(
            ServiceMetadataProvider.iter_objects(
                "flow", "run", None, None, "ExampleFlow"
            )
        )
        == []
    )


def test_get_object_internal_mid_page_404_returns_none(monkeypatch):
    """get_object keeps legacy's atomic contract: a 404 at ANY point in the
    listing resolves to None, never a silently truncated partial list."""
    monkeypatch.setattr(
        ServiceMetadataProvider, "_version", classmethod(_paginating_version)
    )
    calls = []

    def fake_request(cls, monitor, path, method, **kwargs):
        calls.append(path)
        if len(calls) == 1:
            return [_record(2)], {"X-Next-Cursor": "p2", "X-Limit": "1"}
        raise ServiceException("collection deleted mid-scan", http_code=404)

    monkeypatch.setattr(ServiceMetadataProvider, "_request", classmethod(fake_request))

    result = ServiceMetadataProvider._get_object_internal(
        "flow", 1, "run", 2, None, None, "ExampleFlow"
    )
    assert result is None
    assert len(calls) == 2


def test_iter_objects_mid_page_404_ends_stream_with_first_page(monkeypatch):
    """Streaming keeps what was fetched: a mid-pagination 404 just ends the
    stream after the records already yielded."""
    monkeypatch.setattr(
        ServiceMetadataProvider, "_version", classmethod(_paginating_version)
    )
    calls = []

    def fake_request(cls, monitor, path, method, **kwargs):
        calls.append(path)
        if len(calls) == 1:
            return [_record(2)], {"X-Next-Cursor": "p2", "X-Limit": "1"}
        raise ServiceException("collection deleted mid-scan", http_code=404)

    monkeypatch.setattr(ServiceMetadataProvider, "_request", classmethod(fake_request))

    records = list(
        ServiceMetadataProvider.iter_objects("flow", "run", None, None, "ExampleFlow")
    )
    assert [record["run_number"] for record in records] == [2]
    assert len(calls) == 2


def test_get_object_internal_returns_none_when_legacy_fallback_404s(monkeypatch):
    """The no-X-Limit fallback must not mask 'not found' as 'empty': if the
    legacy GET 404s, get_object returns None, not []."""
    monkeypatch.setattr(
        ServiceMetadataProvider, "_version", classmethod(_paginating_version)
    )
    calls = []

    def fake_request(cls, monitor, path, method, **kwargs):
        calls.append(kwargs)
        if "return_headers" in kwargs:
            # Paginated probe: 200 but no X-Limit -> triggers legacy fallback.
            return [_record(9)], {}
        raise ServiceException("collection not found", http_code=404)

    monkeypatch.setattr(ServiceMetadataProvider, "_request", classmethod(fake_request))

    result = ServiceMetadataProvider._get_object_internal(
        "flow", 1, "run", 2, None, None, "ExampleFlow"
    )
    assert result is None
    assert len(calls) == 2


def test_service_paginated_iterator_validates_obj_subtype(monkeypatch):
    """The paginated path enforces the same obj/sub_type guards as get_object."""
    monkeypatch.setattr(
        ServiceMetadataProvider, "_version", classmethod(_paginating_version)
    )

    def unexpected_request(*args, **kwargs):
        raise AssertionError("validation must fail before any request")

    monkeypatch.setattr(
        ServiceMetadataProvider, "_request", classmethod(unexpected_request)
    )

    # 'flow' is not slotted below 'run' -> nonsensical; must raise before any request.
    with pytest.raises(MetaflowInternalError, match="not allowed"):
        list(ServiceMetadataProvider.iter_objects("run", "flow", None, None, "F", "1"))
