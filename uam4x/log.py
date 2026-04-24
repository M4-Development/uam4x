from cpylog import SimpleLogger, _write
import time

DEV = True


def get_log(log=None, filename_log=None, object_name=None):
    """Return a logger with an object-name context layer.

    Parameters
    ----------
    log : FileLogger or LoggerObjectNameLayer, optional
        Existing logger or logger layer to wrap. If omitted, a new
        :class:`FileLogger` is created.
    filename_log : str or path-like, optional
        Log filename to assign when a new root logger is created. This argument
        is ignored when ``log`` is an existing logger instance.
    object_name : str, optional
        Object name to attach to the returned logger layer. If ``log`` is
        already a :class:`LoggerObjectNameLayer` and ``object_name`` is omitted,
        the existing object name is reused. The object name is inserted into
        emitted messages through :class:`LoggerObjectNameLayer`.

    Returns
    -------
    LoggerObjectNameLayer
        Logger wrapper that tags emitted messages with ``object_name``.

    """
    if log is None:
        log = FileLogger()
        log.warning("new log created")
        if filename_log is not None:
            log.set_filename_log(filename_log)

    if isinstance(log, LoggerObjectNameLayer):
        if object_name is None:
            object_name = log.object_name
        log = log.root_logger

    log_with_layer = LoggerObjectNameLayer(log, object_name)
    return log_with_layer


class LoggerObjectNameLayer:
    """Attach an object-name context to a root logger.

    This lightweight wrapper forwards log calls to a root logger while tagging
    each emitted message with the name of the calling object.

    """

    def __init__(self, logger, object_name):
        """Initialize an object-name logger layer.

        Parameters
        ----------
        logger : FileLogger
            Root logger used to emit messages.
        object_name : str or None
            Object name to attach to emitted log messages.

        """
        self.root_logger = logger
        if object_name:
            assert isinstance(object_name, str)
        self.object_name = object_name

    def debug(self, msg):
        """Emit a debug message with the current object-name context.

        Parameters
        ----------
        msg : str
            Log message text.

        Returns
        -------
        None

        """
        self.root_logger.set_next_actor_name(self.object_name)
        self.root_logger.msg_typ("DEBUG", msg)

    def info(self, msg):
        """Emit an info message with the current object-name context.

        Parameters
        ----------
        msg : str
            Log message text.

        Returns
        -------
        None

        """
        self.root_logger.set_next_actor_name(self.object_name)
        self.root_logger.msg_typ("INFO", msg)

    def warning(self, msg):
        """Emit a warning message with the current object-name context.

        Parameters
        ----------
        msg : str
            Log message text.

        Returns
        -------
        None

        """
        self.root_logger.set_next_actor_name(self.object_name)
        self.root_logger.msg_typ("WARNING", msg)

    def error(self, msg):
        """Emit an error message with the current object-name context.

        Parameters
        ----------
        msg : str
            Log message text.

        Returns
        -------
        None

        """
        self.root_logger.set_next_actor_name(self.object_name)
        self.root_logger.msg_typ("ERROR", msg)

    def exception(self, msg):
        """Emit an exception message with the current object-name context.

        Parameters
        ----------
        msg : str
            Log message text.

        Returns
        -------
        None

        """
        self.root_logger.set_next_actor_name(self.object_name)
        self.root_logger.msg_typ("EXCEPTION", msg)

    def critical(self, msg):
        """Emit a critical message with the current object-name context.

        Parameters
        ----------
        msg : str
            Log message text.

        Returns
        -------
        None

        """
        self.root_logger.set_next_actor_name(self.object_name)
        self.root_logger.msg_typ("CRITICAL", msg)


class FileLogger(SimpleLogger):
    """Write UAM4X log messages to stdout, files, and optional GUI handlers.

    ``FileLogger`` extends :class:`cpylog.SimpleLogger` with support for
    file-backed logging, one-shot actor-name context, in-memory capture of
    emitted messages, and optional GUI log forwarding.

    """

    msg_level_dict = {
        "DEBUG": 0,
        "INFO": 1,
        "WARNING": 2,
        "ERROR": 3,
        "EXCEPTION": 4,
        "CRITICAL": 5,
    }

    def __init__(
        self, filename_log=None, write_to_stdout=True, level="debug", encoding="utf-8"
    ):
        """Initialize a file-backed logger.

        Parameters
        ----------
        filename_log : str or path-like, optional
            Log file to open for writing.
        write_to_stdout : bool, optional
            If True, emit log messages to stdout.
        level : str, optional
            Base logger level passed to :class:`cpylog.SimpleLogger`.
        encoding : str, optional
            Text encoding used for stdout and file writes.

        Notes
        -----
        Console and file output thresholds are managed separately through
        :meth:`set_min_console_level` and :meth:`set_min_filelog_level`.

        """
        self.filename_log = filename_log
        self.open_file = open_file(self.filename_log)
        self.write_to_file = self.open_file is not None and not self.open_file.closed

        self.write_to_stdout = write_to_stdout
        self.do_capture_log = False
        self.capture_level_min = "DEBUG"
        # self.capture_level_min = "WARNING"
        self.capture_log = []
        self.gui_log = None
        self.min_filelog_level = "DEBUG"
        self.min_console_level = "INFO"
        self.next_actor_name = None
        super(FileLogger, self).__init__(
            level=level, encoding=encoding, log_func=self.log_function_handler
        )

    def set_gui_log(self, handler):
        """Set a GUI log handler for emitted messages.

        Parameters
        ----------
        handler : object
            GUI-facing handler that receives formatted log messages through an
            ``emit`` method.

        Returns
        -------
        None

        """
        self.gui_log = handler

    def set_next_actor_name(self, actor_name):
        """Set the object-name context for the next emitted message.

        Parameters
        ----------
        actor_name : str
            Name of the object that should be attached to the next emitted log
            message.

        Notes
        -----
        This value is consumed by the next log write and then cleared by the
        log handler.

        """
        self.next_actor_name = actor_name

    def set_min_console_level(self, new_level):
        """Set the minimum message level written to stdout.

        Parameters
        ----------
        new_level : {"DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION", "CRITICAL"}
            Minimum severity level that should appear in console output.
            Messages below this level are still processed internally but are
            not written to stdout.

        Returns
        -------
        None

        """
        assert new_level in self.msg_level_dict
        self.min_console_level = new_level

    def set_min_filelog_level(self, new_level):
        """Set the minimum message level written to the log file.

        Parameters
        ----------
        new_level : {"DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION", "CRITICAL"}
            Minimum severity level that should be written to the active log
            file. Messages below this level are omitted from file output.

        Returns
        -------
        None

        """
        assert new_level in self.msg_level_dict
        self.min_filelog_level = new_level

    def set_filename_log(self, filename_log):
        """Open and attach a log file for subsequent file logging.

        Parameters
        ----------
        filename_log : str or path-like
            Path to the log file to open. This method assumes no log file is
            currently attached.

        Returns
        -------
        None

        Notes
        -----
        The file is opened immediately and subsequent eligible log messages are
        written to it until :meth:`close_file` is called.

        """
        assert self.open_file is None
        self.filename_log = filename_log
        self.open_file = open_file(self.filename_log)
        self.write_to_file = self.open_file is not None and not self.open_file.closed

    def close_file(self):
        """Close the active log file, if one is open.

        Returns
        -------
        None

        Notes
        -----
        After closing the file, file logging is disabled until a new filename
        is attached with :meth:`set_filename_log`.

        """
        if self.open_file:
            self.open_file.close()
        self.open_file = None
        self.write_to_file = False

    def log_function_handler(self, typ, filename, lineno, msg):
        """Route one log message to capture, stdout, and file outputs.

        Parameters
        ----------
        typ : {"DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION", "CRITICAL"}
            Severity level of the emitted message.
        filename : str
            Source filename associated with the log call.
        lineno : int
            Source line number associated with the log call.
        msg : str
            Log message text.

        Returns
        -------
        None

        Notes
        -----
        This is the central dispatch point used by :class:`cpylog.SimpleLogger`.
        Depending on logger configuration, the message may be captured in memory,
        written to stdout, written to the active log file, or any combination of
        those outputs. The one-shot actor context is cleared after dispatch.

        """
        if (
            self.do_capture_log
            and self.msg_level_dict[typ] >= self.msg_level_dict[self.capture_level_min]
        ):
            self.capture_log.append((typ, msg))
        if (
            self.write_to_stdout
            and self.msg_level_dict[typ] >= self.msg_level_dict[self.min_console_level]
        ):
            self.stdout_logging(typ, filename, lineno, msg)
        if (
            self.write_to_file
            and self.open_file is not None
            and not self.open_file.closed
            and self.msg_level_dict[typ] >= self.msg_level_dict[self.min_filelog_level]
        ):
            self.file_logging(typ, filename, lineno, msg)
        self.next_actor_name = None

    def stdout_logging(self, typ: str, filename: str, lineno: int, msg: str) -> None:
        """Write one formatted log message to stdout and any GUI handler.

        Parameters
        ----------
        typ : {"DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION", "CRITICAL"}
            Severity level of the emitted message.
        filename : str
            Source filename associated with the log call.
        lineno : int
            Source line number associated with the log call.
        msg : str
            Log message text.

        Returns
        -------
        None

        Notes
        -----
        The emitted message includes a timestamp and, when available, the current
        actor name. In development mode, filename and line number are included in
        the timestamp portion of the message.

        """
        # max length of 'INFO', 'DEBUG', 'WARNING', etc.
        name = "%-8s" % (typ + ":")
        asctime = time.strftime("%Y-%m-%d %H:%M:%S")
        if DEV:
            asctime = "%s %s:%d" % (asctime, filename, lineno)
        # msg2 = ' %-28s %s\n' % (filename_lineno, msg)
        if self.next_actor_name:
            msg2 = "%s - %s - %s - %s - %s\n" % (
                asctime,
                "uam4x",
                self.next_actor_name,
                typ,
                msg,
            )
        else:
            msg2 = "%s - %s - %s - %s\n" % (asctime, "uam4x", typ, msg)
        # msg2 = "%s - %s - %s - %s\n" % (asctime, "uam4x", typ, msg)

        if self.gui_log:
            self.gui_log.emit(msg2)
        _write(typ, name, msg2, self.encoding)

    def file_logging(self, typ, filename, lineno, msg):
        """Write one formatted log message to the active log file.

        Parameters
        ----------
        typ : {"DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION", "CRITICAL"}
            Severity level of the emitted message.
        filename : str
            Source filename associated with the log call.
        lineno : int
            Source line number associated with the log call.
        msg : str
            Log message text.

        Returns
        -------
        None

        Notes
        -----
        File log messages include the source file and line number, and include the
        current actor name when one has been set.

        """
        # max length of 'INFO', 'DEBUG', 'WARNING', etc.
        name = "%-8s" % (typ + ":")
        filename_n = "%s:%s" % (filename[:-3], lineno)
        if self.next_actor_name:
            msg2 = " %-28s - %s - %s\n" % (filename_n, self.next_actor_name, msg)
        else:
            msg2 = " %-28s %s\n" % (filename_n, msg)

        self.open_file.write((name + msg2) if typ else msg2)
        self.open_file.flush()

    def begin_log_capture(self, capture_level_min="WARNING", clear_capture_log=True):
        """Begin capturing emitted log messages in memory.

        Parameters
        ----------
        capture_level_min : {"DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION", "CRITICAL"}, optional
            Minimum severity level that should be stored in the in-memory
            capture log.
        clear_capture_log : bool, optional
            If True, clear any previously captured messages before beginning a
            new capture session.

        Returns
        -------
        None

        Notes
        -----
        Captured messages are stored as ``(severity, message)`` tuples in
        :attr:`capture_log`.

        """
        assert capture_level_min in self.msg_level_dict
        if self.do_capture_log:
            self.error(
                "Begin Log Capture called without ending the previous log capture"
            )
        else:
            self.do_capture_log = True
            self.capture_level_min = capture_level_min
            if clear_capture_log:
                self.capture_log = []

    def get_capture_log(self):
        """Return the in-memory captured log messages.

        Returns
        -------
        list of tuple[str, str]
            Captured ``(severity, message)`` pairs accumulated since capture began
            or was last cleared.

        """
        return self.capture_log

    def end_log_capture(self):
        """End in-memory log capture and return the captured messages.

        Returns
        -------
        list of tuple[str, str]
            Captured ``(severity, message)`` pairs accumulated during the capture
            session.

        Notes
        -----
        This method disables further capture but does not clear the stored
        messages.

        """
        self.do_capture_log = False
        return self.capture_log

    def __del__(self):
        """Close the active log file during object cleanup.

        Returns
        -------
        None

        """
        if self.open_file is not None and not self.open_file.closed:
            self.open_file.close()

    def __repr__(self):
        """Return a developer-oriented string representation of the logger.

        Returns
        -------
        str
            Representation including the configured log filename, stdout setting,
            logger level, and encoding.

        """
        return (
            "FileLogger(filename_log=%r, write_to_stdout=%r, level=%r, encoding=%r)"
            % (self.filename_log, self.write_to_file, self.level, self.encoding)
        )


def open_file(filename_log):
    """Open a log file for writing if a filename is provided.

    Parameters
    ----------
    filename_log : str or path-like
        Path to the log file to open.

    Returns
    -------
    file object or None
        Open file handle if ``filename_log`` is provided, otherwise None.

    """
    open_file = None
    if filename_log is not None:
        open_file = open(filename_log, "w")
    return open_file
