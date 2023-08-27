"""Database wrapper."""

import asyncio
from concurrent import futures
from functools import partial
from itertools import chain
import logging
import sys
from threading import (
    current_thread,
    Thread,
)
from time import perf_counter
from traceback import format_tb
from typing import (
    Any,
    Callable,
    cast,
    NamedTuple,
)

from croniter import croniter
from sqlalchemy import (
    create_engine,
    event,
    text,
)
from sqlalchemy.engine import (
    Connection,
    Engine,
    Result,
)
from sqlalchemy.exc import (
    ArgumentError,
    NoSuchModuleError,
)
from sqlalchemy.sql.elements import TextClause

#: Timeout for a query
QueryTimeout = int | float


#: Label used to tag metrics by database
DATABASE_LABEL = "database"


class DataBaseError(Exception):
    """A databease error.

    if `fatal` is True, it means the Query will never succeed.
    """

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


class DataBaseConnectError(DataBaseError):
    """Database connection error."""


class DataBaseQueryError(DataBaseError):
    """Database query error."""


class QueryTimeoutExpired(Exception):
    """Query execution timeout expired."""

    def __init__(self, query_name: str, timeout: QueryTimeout):
        super().__init__(
            f'Execution for query "{query_name}" expired after {timeout} seconds'
        )


class InvalidResultCount(Exception):
    """Number of results from a query don't match metrics count."""

    def __init__(self, expected: int, got: int):
        super().__init__(
            f"Wrong result count from query: expected {expected}, got {got}"
        )


class InvalidResultColumnNames(Exception):
    """Invalid column names in query results."""

    def __init__(self, expected: list[str], got: list[str]) -> None:
        super().__init__(
            "Wrong column names from query: "
            f"expected {self._names(expected)}, got {self._names(got)}"
        )

    def _names(self, names: list[str]) -> str:
        names_list = ", ".join(names)
        return f"({names_list})"


class InvalidQueryParameters(Exception):
    """Query parameter names don't match those in query SQL."""

    def __init__(self, query_name: str):
        super().__init__(
            f'Parameters for query "{query_name}" don\'t match those from SQL'
        )


class InvalidQuerySchedule(Exception):
    """Query schedule is wrong or both schedule and interval specified."""

    def __init__(self, query_name: str, message: str):
        super().__init__(
            f'Invalid schedule for query "{query_name}": {message}'
        )


# Database errors that mean the query won't ever succeed.  Not all possible
# fatal errors are tracked here, because some DBAPI errors can happen in
# circumstances which can be fatal or not.  Since there doesn't seem to be a
# reliable way to know, there might be cases when a query will never succeed
# but will end up being retried.
FATAL_ERRORS = (InvalidResultCount, InvalidResultColumnNames)


def create_db_engine(dsn: str, **kwargs) -> Engine:
    """Create the database engine, validating the DSN"""
    try:
        return create_engine(dsn, **kwargs)
    except ImportError as error:
        raise DataBaseError(f'module "{error.name}" not found')
    except (ArgumentError, ValueError, NoSuchModuleError):
        raise DataBaseError(f'Invalid database DSN: "{dsn}"')


class QueryMetric(NamedTuple):
    """Metric details for a Query."""

    name: str
    labels: list[str]


class QueryResults(NamedTuple):
    """Results of a database query."""

    keys: list[str]
    rows: list[tuple]
    latency: float | None = None

    @classmethod
    def from_result(cls, result: Result) -> "QueryResults":
        """Return a QueryResults from results for a query."""
        keys, rows = [], []
        if result.returns_rows:
            keys, rows = list(result.keys()), result.all()
        latency = result.connection.info.get("query_latency", None)
        return cls(keys, rows, latency=latency)


class MetricResult(NamedTuple):
    """A result for a metric from a query."""

    metric: str
    value: Any
    labels: dict[str, str]


class MetricResults(NamedTuple):
    """Collection of metric results for a query."""

    results: list[MetricResult]
    latency: float | None = None


class Query:
    """Query definition and configuration."""

    def __init__(
        self,
        name: str,
        databases: list[str],
        metrics: list[QueryMetric],
        sql: str,
        parameters: dict[str, Any] | None = None,
        timeout: QueryTimeout | None = None,
        interval: int | None = None,
        schedule: str | None = None,
        config_name: str | None = None,
    ):
        self.name = name
        self.databases = databases
        self.metrics = metrics
        self.sql = sql
        self.parameters = parameters or {}
        self.timeout = timeout
        self.interval = interval
        self.schedule = schedule
        self.config_name = config_name or name
        self._check_schedule()
        self._check_query_parameters()

    @property
    def timed(self) -> bool:
        """Whether the query is run periodically via interval or schedule."""
        return bool(self.interval or self.schedule)

    def labels(self) -> frozenset[str]:
        """Resturn all labels for metrics in the query."""
        return frozenset(chain(*(metric.labels for metric in self.metrics)))

    def results(self, query_results: QueryResults) -> MetricResults:
        """Return MetricResults from a query."""
        if not query_results.rows:
            return MetricResults([])

        result_keys = sorted(query_results.keys)
        labels = self.labels()
        metrics = [metric.name for metric in self.metrics]
        expected_keys = sorted(set(metrics) | labels)
        if len(expected_keys) != len(result_keys):
            raise InvalidResultCount(len(expected_keys), len(result_keys))
        if result_keys != expected_keys:
            raise InvalidResultColumnNames(result_keys, expected_keys)
        results = []
        for row in query_results.rows:
            values = dict(zip(query_results.keys, row))
            for metric in self.metrics:
                metric_result = MetricResult(
                    metric.name,
                    values[metric.name],
                    {label: values[label] for label in metric.labels},
                )
                results.append(metric_result)

        return MetricResults(results, latency=query_results.latency)

    def _check_schedule(self):
        if self.interval and self.schedule:
            raise InvalidQuerySchedule(
                self.name, "both interval and schedule specified"
            )
        if self.schedule and not croniter.is_valid(self.schedule):
            raise InvalidQuerySchedule(self.name, "invalid schedule format")

    def _check_query_parameters(self):
        expr = text(self.sql)
        query_params = set(expr.compile().params)
        if set(self.parameters) != query_params:
            raise InvalidQueryParameters(self.name)


class WorkerAction:
    """An action to be called in the worker thread."""

    def __init__(self, func: Callable):
        self.func = func
        self._future: asyncio.Future = asyncio.get_event_loop().create_future()

    def __str__(self) -> str:
        if isinstance(self.func, partial):
            return self.func.func.__name__
        return self.func.__name__

    def set_result(self, result):
        """Set the result of the action."""
        self._future.set_result(result)

    def set_exception(self, exception: Exception):
        """Set the result of the action."""
        self._future.set_exception(exception)

    async def result(self):
        """Return the action result."""
        return await self._future


class DataBaseConnection:
    """A connection to a database engine."""

    _conn: Connection | None = None
    _worker: Thread | None = None

    def __init__(
        self,
        dbname: str,
        engine: Engine,
        logger: logging.Logger = logging.getLogger(),
    ):
        self.dbname = dbname
        self.engine = engine
        self.logger = logger
        self._loop = asyncio.get_event_loop()
        self._queue: asyncio.Queue = asyncio.Queue()

    @property
    def connected(self) -> bool:
        """Whether the connection is open."""
        return self._conn is not None

    async def open(self):
        """Open the connection."""
        if self.connected:
            return

        self._create_worker()
        return await self._call_in_thread(self._connect)

    async def close(self):
        """Close the connection."""
        if not self.connected:
            return

        await self._call_in_thread(self._close)

    async def execute(
        self,
        sql: TextClause,
        parameters: dict[str, Any] | None = None,
    ) -> QueryResults:
        """Execute a query, returning results."""
        if parameters is None:
            parameters = {}
        result = await self._call_in_thread(self._execute, sql, parameters)
        query_results: QueryResults = await self._call_in_thread(
            QueryResults.from_result, result
        )
        return query_results

    def _create_worker(self):
        """Create the worker thread."""
        self._worker = Thread(
            target=self._run, name=f"DataBase-{self.dbname}", daemon=True
        )
        self._worker.start()

    def _connect(self):
        self._conn = self.engine.connect()

    def _execute(self, sql: TextClause, parameters: dict[str, Any]) -> Result:
        self._conn: Connection
        return self._conn.execute(sql, parameters)

    def _close(self):
        self._conn: Connection
        self._conn.detach()
        self._conn.close()
        self._conn = None

    def _run(self):
        """The worker thread function."""

        def debug(message: str):
            self.logger.debug(f'worker "{current_thread().name}": {message}')

        debug(f"started with ID {current_thread().native_id}")
        while True:
            future = asyncio.run_coroutine_threadsafe(
                self._queue.get(), self._loop
            )
            try:
                action = future.result()
                debug(f'received action "{action}"')
            except futures.CancelledError:
                debug("shutting down")
                return

            try:
                result = action.func()
            except Exception as e:
                self._loop.call_soon_threadsafe(action.set_exception, e)
            else:
                self._loop.call_soon_threadsafe(action.set_result, result)
            finally:
                self._loop.call_soon_threadsafe(self._queue.task_done)
                if self._conn is None:
                    # the connection has been closed, exit the thread
                    debug("shutting down")
                    return

    async def _call_in_thread(self, func: Callable, *args, **kwargs):
        """Call a sync action in the worker thread."""
        call = WorkerAction(partial(func, *args, **kwargs))
        await self._queue.put(call)
        return await call.result()


class DataBase:
    """A database to perform Queries."""

    _conn: DataBaseConnection
    _logger: logging.Logger = logging.getLogger()
    _pending_queries: int = 0

    def __init__(
        self,
        config,
        logger: logging.Logger = logging.getLogger(),
    ):
        self.config = config
        self.logger = logger
        self._connect_lock = asyncio.Lock()
        engine = create_db_engine(
            self.config.dsn,
            execution_options={"autocommit": self.config.autocommit},
        )
        self._conn = DataBaseConnection(self.config.name, engine, self.logger)
        self._setup_query_latency_tracking(engine)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    @property
    def connected(self) -> bool:
        """Whether the database is connected."""
        return self._conn.connected

    async def connect(self):
        """Connect to the database."""
        async with self._connect_lock:
            if self.connected:
                return

            try:
                await self._conn.open()
            except Exception as error:
                raise self._db_error(error, exc_class=DataBaseConnectError)

            self.logger.debug(f'connected to database "{self.config.name}"')
            for sql in self.config.connect_sql:
                try:
                    await self.execute_sql(sql)
                except Exception as error:
                    await self._close()
                    raise self._db_error(
                        f'failed executing query "{sql}": {error}',
                        exc_class=DataBaseQueryError,
                    )

    async def close(self):
        """Close the database connection."""
        async with self._connect_lock:
            if not self.connected:
                return
            await self._close()

    async def execute(self, query: Query) -> MetricResults:
        """Execute a query."""
        await self.connect()
        self.logger.debug(
            f'running query "{query.name}" on database "{self.config.name}"'
        )
        self._pending_queries += 1
        try:
            query_results = await self.execute_sql(
                query.sql, parameters=query.parameters, timeout=query.timeout
            )
            return query.results(query_results)
        except asyncio.TimeoutError:
            raise self._query_timeout_error(
                query.name, cast(QueryTimeout, query.timeout)
            )
        except Exception as error:
            raise self._query_db_error(
                query.name, error, fatal=isinstance(error, FATAL_ERRORS)
            )
        finally:
            assert self._pending_queries >= 0, "pending queries is negative"
            self._pending_queries -= 1
            if not self.config.keep_connected and not self._pending_queries:
                await self.close()

    async def execute_sql(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
        timeout: QueryTimeout | None = None,
    ) -> QueryResults:
        """Execute a raw SQL query."""
        if parameters is None:
            parameters = {}
        return await asyncio.wait_for(
            self._conn.execute(text(sql), parameters),
            timeout=timeout,
        )

    async def _close(self):
        # ensure the connection with the DB is actually closed
        await self._conn.close()
        self._pending_queries = 0
        self.logger.debug(f'disconnected from database "{self.config.name}"')

    def _setup_query_latency_tracking(self, engine: Engine):
        @event.listens_for(engine, "before_cursor_execute")
        def before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            conn.info["query_start_time"] = perf_counter()

        @event.listens_for(engine, "after_cursor_execute")
        def after_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            conn.info["query_latency"] = perf_counter() - conn.info.pop(
                "query_start_time"
            )

    def _query_db_error(
        self,
        query_name: str,
        error: str | Exception,
        fatal: bool = False,
    ) -> DataBaseError:
        """Create and log a DataBaseError for a failed query."""
        message = self._error_message(error)
        self.logger.error(
            f'query "{query_name}" on database "{self.config.name}" failed: '
            + message
        )
        _, _, traceback = sys.exc_info()
        self.logger.debug("".join(format_tb(traceback)))
        return DataBaseQueryError(message, fatal=fatal)

    def _query_timeout_error(
        self, query_name: str, timeout: QueryTimeout
    ) -> QueryTimeoutExpired:
        error = QueryTimeoutExpired(query_name, timeout)
        self.logger.warning(str(error))
        raise error

    def _db_error(
        self,
        error: str | Exception,
        exc_class: type[DataBaseError] = DataBaseError,
        fatal: bool = False,
    ) -> DataBaseError:
        """Create and log a DataBaseError."""
        message = self._error_message(error)
        self.logger.error(
            f'error from database "{self.config.name}": {message}'
        )
        return exc_class(message, fatal=fatal)

    def _error_message(self, error: str | Exception) -> str:
        """Return a message from an error."""
        message = str(error).strip()
        if not message and isinstance(error, Exception):
            message = error.__class__.__name__
        return message
