import pytest
import uuid
from unittest.mock import MagicMock, patch, call, PropertyMock
from postgrest import APIError, APIResponse


# ---------------------------------------------------------------------------
# Helpers & shared constants
# ---------------------------------------------------------------------------

FAKE_UUID = "11111111-1111-1111-1111-111111111111"
EMPTY_UUID = "00000000-0000-0000-0000-000000000000"
FAKE_FILENAME = "test_document.pdf"
FAKE_DIM = 384  # typical sentence-transformer embedding size

def make_api_response(data: list) -> MagicMock:
    """Return a MagicMock that quacks like a postgrest APIResponse."""
    resp = MagicMock(spec=APIResponse)
    resp.data = data
    return resp

def make_doc_record(filename: str = FAKE_FILENAME, doc_uuid: str = FAKE_UUID) -> dict:
    return {"filename": filename, "uuid": doc_uuid, "title": "Test Doc"}

def make_chunk_records(n: int = 3, dim: int = FAKE_DIM) -> list:
    return [
        (f"chunk-{i}", [0.1] * dim, {"text": f"chunk text {i}"})
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Fixtures: patch supabase client and vecs at module level
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Ensure env vars are always set so db.py can import cleanly."""
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake-key")
    monkeypatch.setenv("POOLER_URL", "postgresql://fake:fake@fake/fake")


@pytest.fixture()
def mock_supabase_client():
    """
    Patches `db.supabase` (the already-created client) so no real network
    calls are made.  Returns the mock so individual tests can configure it.
    """
    with patch("db.supabase") as mock_client:
        # Build a fluent query-builder chain:
        # supabase.table(...).select(...).eq(...).execute()
        query_builder = MagicMock()
        query_builder.select.return_value = query_builder
        query_builder.insert.return_value = query_builder
        query_builder.delete.return_value = query_builder
        query_builder.eq.return_value = query_builder

        mock_client.table.return_value = query_builder
        mock_client._query_builder = query_builder  # convenience handle
        yield mock_client


@pytest.fixture()
def mock_vecs():
    """Patches vecs.create_client so no real DB connection is made."""
    with patch("db.vecs") as mock_vecs_module:
        mock_collection = MagicMock()
        mock_vx = MagicMock()
        mock_vx.get_or_create_collection.return_value = mock_collection
        mock_vecs_module.create_client.return_value = mock_vx
        mock_vecs_module.IndexMeasure.cosine_distance = "cosine_distance"
        yield mock_vecs_module, mock_collection


# ---------------------------------------------------------------------------
# Tests: doc_check_exists
# ---------------------------------------------------------------------------

class TestDocCheckExists:

    def test_check_exists_found_by_filename(self, mock_supabase_client):
        """Returns a response with data when the document is found."""
        from db import doc_check_exists

        mock_supabase_client._query_builder.execute.return_value = make_api_response(
            [{"filename": FAKE_FILENAME}]
        )

        result = doc_check_exists("filename", FAKE_FILENAME)

        assert result.data == [{"filename": FAKE_FILENAME}]
        mock_supabase_client.table.assert_called_once_with("documents")

    def test_check_exists_found_by_uuid(self, mock_supabase_client):
        """Returns a response with data when searching by uuid."""
        from db import doc_check_exists

        mock_supabase_client._query_builder.execute.return_value = make_api_response(
            [{"uuid": FAKE_UUID}]
        )

        result = doc_check_exists("uuid", FAKE_UUID)

        assert result.data == [{"uuid": FAKE_UUID}]

    def test_check_exists_not_found(self, mock_supabase_client):
        """Returns a response with empty data when document is not found."""
        from db import doc_check_exists

        mock_supabase_client._query_builder.execute.return_value = make_api_response([])

        result = doc_check_exists("filename", "nonexistent.pdf")

        assert result.data == []

    def test_check_exists_api_error_returns_minus_one(self, mock_supabase_client):
        """Returns -1 when an APIError is raised."""
        from db import doc_check_exists

        mock_supabase_client._query_builder.execute.side_effect = APIError(
            {"message": "connection error", "code": "500", "details": "", "hint": ""}
        )

        result = doc_check_exists("filename", FAKE_FILENAME)

        assert result == -1

    def test_check_exists_calls_correct_column_and_value(self, mock_supabase_client):
        """Verifies the query is built with the correct column and value."""
        from db import doc_check_exists

        qb = mock_supabase_client._query_builder
        qb.execute.return_value = make_api_response([])

        doc_check_exists("uuid", FAKE_UUID)

        qb.select.assert_called_once_with("uuid")
        qb.eq.assert_called_once_with("uuid", FAKE_UUID)


# ---------------------------------------------------------------------------
# Tests: doc_insert
# ---------------------------------------------------------------------------

class TestDocInsert:

    def test_insert_success(self, mock_supabase_client):
        """Inserts a new document and returns the APIResponse."""
        from db import doc_insert

        qb = mock_supabase_client._query_builder
        # First call (check_exists): empty → doc doesn't exist
        # Second call (insert): returns the new row
        qb.execute.side_effect = [
            make_api_response([]),                              # check_exists
            make_api_response([make_doc_record()]),            # insert
        ]

        result = doc_insert(FAKE_FILENAME, make_doc_record())

        assert result.data == [make_doc_record()]

    def test_insert_duplicate_returns_minus_one(self, mock_supabase_client):
        """Returns -1 when the document already exists."""
        from db import doc_insert

        qb = mock_supabase_client._query_builder
        qb.execute.return_value = make_api_response([{"filename": FAKE_FILENAME}])

        result = doc_insert(FAKE_FILENAME, make_doc_record())

        assert result == -1

    def test_insert_check_exists_error_returns_minus_one(self, mock_supabase_client):
        """Returns -1 when doc_check_exists itself errors."""
        from db import doc_insert

        qb = mock_supabase_client._query_builder
        qb.execute.side_effect = APIError(
            {"message": "db error", "code": "500", "details": "", "hint": ""}
        )

        result = doc_insert(FAKE_FILENAME, make_doc_record())

        assert result == -1

    def test_insert_api_error_on_insert_returns_minus_one(self, mock_supabase_client):
        """Returns -1 when the actual INSERT call raises an APIError."""
        from db import doc_insert

        qb = mock_supabase_client._query_builder
        qb.execute.side_effect = [
            make_api_response([]),   # check_exists → doc does not exist
            APIError({"message": "constraint violation", "code": "409", "details": "", "hint": ""}),
        ]

        result = doc_insert(FAKE_FILENAME, make_doc_record())

        assert result == -1

    def test_insert_calls_insert_with_correct_record(self, mock_supabase_client):
        """Verifies that .insert() is called with the exact record dict."""
        from db import doc_insert

        qb = mock_supabase_client._query_builder
        record = make_doc_record()
        qb.execute.side_effect = [
            make_api_response([]),
            make_api_response([record]),
        ]

        doc_insert(FAKE_FILENAME, record)

        qb.insert.assert_called_once_with(record)


# ---------------------------------------------------------------------------
# Tests: doc_delete
# ---------------------------------------------------------------------------

class TestDocDelete:

    def test_delete_by_filename_success(self, mock_supabase_client):
        """Deletes a document by filename and returns the APIResponse."""
        from db import doc_delete

        qb = mock_supabase_client._query_builder
        qb.execute.return_value = make_api_response([{"filename": FAKE_FILENAME}])

        result = doc_delete("filename", FAKE_FILENAME)

        assert result.data == [{"filename": FAKE_FILENAME}]
        qb.delete.assert_called_once()
        qb.eq.assert_called_once_with("filename", FAKE_FILENAME)

    def test_delete_by_uuid_success(self, mock_supabase_client):
        """Deletes a document by uuid and returns the APIResponse."""
        from db import doc_delete

        qb = mock_supabase_client._query_builder
        qb.execute.return_value = make_api_response([{"uuid": FAKE_UUID}])

        result = doc_delete("uuid", FAKE_UUID)

        assert result.data == [{"uuid": FAKE_UUID}]

    def test_delete_nonexistent_document(self, mock_supabase_client):
        """Returns an empty response (not -1) when no row matched."""
        from db import doc_delete

        qb = mock_supabase_client._query_builder
        qb.execute.return_value = make_api_response([])

        result = doc_delete("filename", "ghost.pdf")

        assert result.data == []

    def test_delete_api_error_returns_minus_one(self, mock_supabase_client):
        """Returns -1 when an APIError is raised during delete."""
        from db import doc_delete

        qb = mock_supabase_client._query_builder
        qb.execute.side_effect = APIError(
            {"message": "permission denied", "code": "403", "details": "", "hint": ""}
        )

        result = doc_delete("filename", FAKE_FILENAME)

        assert result == -1


# ---------------------------------------------------------------------------
# Tests: vec_init_db
# ---------------------------------------------------------------------------

class TestVecInitDb:

    def test_init_db_returns_collection(self, mock_vecs):
        """Returns the vecs Collection object on success."""
        from db import vec_init_db

        mock_vecs_module, mock_collection = mock_vecs

        result = vec_init_db(FAKE_DIM)

        assert result is mock_collection

    def test_init_db_creates_collection_with_correct_dimension(self, mock_vecs):
        """Calls get_or_create_collection with name='chunks' and the given dim."""
        from db import vec_init_db

        mock_vecs_module, mock_collection = mock_vecs
        mock_vx = mock_vecs_module.create_client.return_value

        vec_init_db(FAKE_DIM)

        mock_vx.get_or_create_collection.assert_called_once_with(
            name="chunks", dimension=FAKE_DIM
        )

    def test_init_db_creates_index(self, mock_vecs):
        """Calls create_index on the returned collection."""
        from db import vec_init_db

        mock_vecs_module, mock_collection = mock_vecs

        vec_init_db(FAKE_DIM)

        mock_collection.create_index.assert_called_once()

    def test_init_db_uses_pooler_url(self, mock_vecs):
        """vecs.create_client is called with POOLER_URL from env."""
        from db import vec_init_db
        import os

        mock_vecs_module, _ = mock_vecs
        expected_url = os.getenv("POOLER_URL")

        vec_init_db(FAKE_DIM)

        mock_vecs_module.create_client.assert_called_once_with(expected_url)

    def test_init_db_different_dimension(self, mock_vecs):
        """Works with any positive integer dimension."""
        from db import vec_init_db

        mock_vecs_module, _ = mock_vecs
        mock_vx = mock_vecs_module.create_client.return_value

        vec_init_db(1536)

        mock_vx.get_or_create_collection.assert_called_once_with(
            name="chunks", dimension=1536
        )


# ---------------------------------------------------------------------------
# Tests: vec_batch_upsert
# ---------------------------------------------------------------------------

class TestVecBatchUpsert:

    def test_upsert_success_returns_none(self, mock_supabase_client):
        """Returns None (implicitly) when document does NOT yet exist."""
        from db import vec_batch_upsert

        qb = mock_supabase_client._query_builder
        qb.execute.return_value = make_api_response([])  # doc not found

        mock_collection = MagicMock()
        records = make_chunk_records(5)

        result = vec_batch_upsert(mock_collection, records, EMPTY_UUID)

        assert result is None

    def test_upsert_calls_collection_upsert(self, mock_supabase_client):
        """Calls collection.upsert() at least once with record batches."""
        from db import vec_batch_upsert

        qb = mock_supabase_client._query_builder
        qb.execute.return_value = make_api_response([])

        mock_collection = MagicMock()
        records = make_chunk_records(3)

        vec_batch_upsert(mock_collection, records, EMPTY_UUID, batch_size=500)

        mock_collection.upsert.assert_called_once_with(records=records)

    def test_upsert_batches_correctly(self, mock_supabase_client):
        """Splits records into correct number of batches."""
        from db import vec_batch_upsert

        qb = mock_supabase_client._query_builder
        qb.execute.return_value = make_api_response([])

        mock_collection = MagicMock()
        records = make_chunk_records(10)

        vec_batch_upsert(mock_collection, records, EMPTY_UUID, batch_size=3)

        # 10 records / batch_size 3 → ceil(10/3) = 4 upsert calls
        assert mock_collection.upsert.call_count == 4

    def test_upsert_correct_batch_contents(self, mock_supabase_client):
        """Each upsert call receives the correct slice of records."""
        from db import vec_batch_upsert

        qb = mock_supabase_client._query_builder
        qb.execute.return_value = make_api_response([])

        mock_collection = MagicMock()
        records = make_chunk_records(5)

        vec_batch_upsert(mock_collection, records, EMPTY_UUID, batch_size=2)

        calls = mock_collection.upsert.call_args_list
        assert calls[0] == call(records=records[0:2])
        assert calls[1] == call(records=records[2:4])
        assert calls[2] == call(records=records[4:5])

    def test_upsert_document_already_exists_returns_zero(self, mock_supabase_client):
        """Returns 0 without upserting when the document already exists."""
        from db import vec_batch_upsert

        qb = mock_supabase_client._query_builder
        qb.execute.return_value = make_api_response([{"uuid": FAKE_UUID}])

        mock_collection = MagicMock()
        records = make_chunk_records(3)

        result = vec_batch_upsert(mock_collection, records, FAKE_UUID)

        assert result == 0
        mock_collection.upsert.assert_not_called()

    def test_upsert_check_exists_error_returns_minus_one(self, mock_supabase_client):
        """Returns -1 when doc_check_exists encounters an APIError."""
        from db import vec_batch_upsert

        qb = mock_supabase_client._query_builder
        qb.execute.side_effect = APIError(
            {"message": "server error", "code": "500", "details": "", "hint": ""}
        )

        mock_collection = MagicMock()

        result = vec_batch_upsert(mock_collection, make_chunk_records(), FAKE_UUID)

        assert result == -1
        mock_collection.upsert.assert_not_called()

    def test_upsert_empty_records_list(self, mock_supabase_client):
        """Handles an empty records list without calling upsert."""
        from db import vec_batch_upsert

        qb = mock_supabase_client._query_builder
        qb.execute.return_value = make_api_response([])

        mock_collection = MagicMock()

        result = vec_batch_upsert(mock_collection, [], EMPTY_UUID)

        assert result is None
        mock_collection.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: upload_new_doc
# ---------------------------------------------------------------------------

class TestUploadNewDoc:

    def _make_mock_collection(self):
        return MagicMock()

    def test_upload_success(self, mock_supabase_client, mock_vecs):
        """Full happy path returns None (success)."""
        from db import upload_new_doc

        qb = mock_supabase_client._query_builder
        record = make_doc_record()
        # check_exists (doc_insert) → empty; insert → success; check_exists (vec_upsert) → empty
        qb.execute.side_effect = [
            make_api_response([]),           # doc_check_exists inside doc_insert
            make_api_response([record]),     # doc_insert itself
            make_api_response([]),           # doc_check_exists inside vec_batch_upsert
        ]

        _, mock_collection = mock_vecs
        chunks_records = make_chunk_records(3)

        result = upload_new_doc(FAKE_FILENAME, record, chunks_records)

        assert result is None
        mock_collection.upsert.assert_called_once()

    def test_upload_doc_insert_failure_returns_minus_one(self, mock_supabase_client, mock_vecs):
        """Returns -1 immediately when doc_insert fails."""
        from db import upload_new_doc

        qb = mock_supabase_client._query_builder
        # check_exists → doc already exists → doc_insert returns -1
        qb.execute.return_value = make_api_response([{"filename": FAKE_FILENAME}])

        _, mock_collection = mock_vecs

        result = upload_new_doc(FAKE_FILENAME, make_doc_record(), make_chunk_records())

        assert result == -1
        mock_collection.upsert.assert_not_called()

    def test_upload_vec_upsert_failure_triggers_doc_delete(self, mock_supabase_client, mock_vecs):
        """When vec_batch_upsert fails, it rolls back by deleting the document."""
        from db import upload_new_doc

        qb = mock_supabase_client._query_builder
        record = make_doc_record()

        # Sequence:
        # 1. doc_check_exists (inside doc_insert) → empty (OK)
        # 2. doc_insert execute → success
        # 3. doc_check_exists (inside vec_batch_upsert) → APIError → returns -1
        # 4. doc_delete execute (rollback) → success
        qb.execute.side_effect = [
            make_api_response([]),           # 1
            make_api_response([record]),     # 2
            APIError({"message": "err", "code": "500", "details": "", "hint": ""}),  # 3
            make_api_response([record]),     # 4 rollback delete
        ]

        _, mock_collection = mock_vecs

        result = upload_new_doc(FAKE_FILENAME, record, make_chunk_records())

        assert result == -1
        # The delete (rollback) must be called with the doc's uuid
        qb.delete.assert_called_once()

    def test_upload_uses_empty_uuid_for_vec_check(self, mock_supabase_client, mock_vecs):
        """
        vec_batch_upsert is always called with the empty UUID so the
        'document already exists' guard is bypassed for new chunks.
        """
        from db import upload_new_doc

        qb = mock_supabase_client._query_builder
        record = make_doc_record()
        qb.execute.side_effect = [
            make_api_response([]),
            make_api_response([record]),
            make_api_response([]),
        ]

        _, mock_collection = mock_vecs

        with patch("db.vec_batch_upsert", wraps=None) as mock_vbu:
            mock_vbu.return_value = None

            # Re-import to pick up the patch — use a targeted patch instead
            pass

        # Verify indirectly: the check_exists inside vec_batch_upsert is called with
        # the empty UUID by checking the .eq() call args list.
        upload_new_doc(FAKE_FILENAME, record, make_chunk_records())

        eq_calls = [str(c) for c in qb.eq.call_args_list]
        # One of the .eq() calls must use the empty UUID
        assert any(EMPTY_UUID in c for c in eq_calls)

    def test_upload_passes_batch_size_through(self, mock_supabase_client, mock_vecs):
        """Custom batch_size is forwarded to vec_batch_upsert."""
        from db import upload_new_doc

        qb = mock_supabase_client._query_builder
        record = make_doc_record()
        chunks = make_chunk_records(10)

        qb.execute.side_effect = [
            make_api_response([]),
            make_api_response([record]),
            make_api_response([]),
        ]

        _, mock_collection = mock_vecs

        upload_new_doc(FAKE_FILENAME, record, chunks, batch_size=2)

        # 10 records / batch 2 → 5 upsert calls
        assert mock_collection.upsert.call_count == 5

    def test_upload_doc_insert_api_error_returns_minus_one(self, mock_supabase_client, mock_vecs):
        """Returns -1 when doc_insert raises an APIError on the INSERT call."""
        from db import upload_new_doc

        qb = mock_supabase_client._query_builder
        qb.execute.side_effect = [
            make_api_response([]),  # check_exists → ok
            APIError({"message": "insert failed", "code": "500", "details": "", "hint": ""}),
        ]

        _, mock_collection = mock_vecs

        result = upload_new_doc(FAKE_FILENAME, make_doc_record(), make_chunk_records())

        assert result == -1
        mock_collection.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# Integration tests (skipped unless INTEGRATION env var is set)
# ---------------------------------------------------------------------------

integration = pytest.mark.skipif(
    not pytest.importorskip("os").environ.get("INTEGRATION"),
    reason="Set INTEGRATION=1 to run integration tests against real Supabase",
)


@integration
class TestIntegration:
    """
    Runs against a real Supabase instance. Cleans up all created rows after
    each test via a yield fixture.

    Requires: SUPABASE_URL, SUPABASE_KEY, POOLER_URL in environment.
    """

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Delete the test document after each test regardless of outcome."""
        yield
        # Teardown: remove any residual test rows by filename and uuid
        import db
        db.doc_delete("filename", FAKE_FILENAME)
        db.doc_delete("uuid", FAKE_UUID)

    def test_integration_insert_and_check_exists(self):
        import db

        record = make_doc_record()
        insert_result = db.doc_insert(FAKE_FILENAME, record)
        assert insert_result != -1
        assert len(insert_result.data) == 1

        exists_result = db.doc_check_exists("filename", FAKE_FILENAME)
        assert exists_result != -1
        assert len(exists_result.data) == 1

    def test_integration_duplicate_insert_rejected(self):
        import db

        record = make_doc_record()
        db.doc_insert(FAKE_FILENAME, record)
        second = db.doc_insert(FAKE_FILENAME, record)
        assert second == -1

    def test_integration_delete_removes_document(self):
        import db

        record = make_doc_record()
        db.doc_insert(FAKE_FILENAME, record)

        db.doc_delete("filename", FAKE_FILENAME)

        exists = db.doc_check_exists("filename", FAKE_FILENAME)
        assert len(exists.data) == 0

    def test_integration_check_nonexistent_returns_empty(self):
        import db

        result = db.doc_check_exists("filename", "definitely_does_not_exist.pdf")
        assert result != -1
        assert result.data == []

    def test_integration_upload_new_doc_end_to_end(self):
        import db

        record = make_doc_record()
        chunks = make_chunk_records(5, dim=db.SENTENCETRANSFORMEREMBEDDINGSIZE)

        result = db.upload_new_doc(FAKE_FILENAME, record, chunks)
        assert result is None

        # Verify the document row was persisted
        exists = db.doc_check_exists("filename", FAKE_FILENAME)
        assert len(exists.data) == 1