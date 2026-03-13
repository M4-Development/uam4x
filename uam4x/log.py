from cpylog import SimpleLogger, _write
import time

DEV = True


def get_log(log=None, filename_log=None, object_name=None):
    """
    Returns a a new log if none given, or returns the log given with an appropriate object_name.

    Parameters
    ----------
    log : LoggerObjectNameLayer
        The root log with an object name layer on top of it.
    filename_log : str
        If a new log is generated, filename_log will be used as the log filename
    object_name : str
        The name of the object this method was called in. If none given, will use previous log's object name.

    Returns
    --------
    log_with_layer : LoggerObjectNameLayer
        Either a new log, the same log, or the same log with a different object name.

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
    """Represent the loggerobjectnamelayer component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """

    def __init__(self, logger, object_name):
        """Initialize object state and default configuration.

        Parameters
        ----------
        logger : Any
            Input value for `logger`.
        object_name : Any
            Identifier used by this operation.

        """
        self.root_logger = logger
        if object_name:
            assert isinstance(object_name, str)
        self.object_name = object_name

    def debug(self, msg):
        """Execute debug.

        Parameters
        ----------
        msg : Any
            Input value for `msg`.

        """
        self.root_logger.set_next_actor_name(self.object_name)
        self.root_logger.msg_typ("DEBUG", msg)

    def info(self, msg):
        """Execute info.

        Parameters
        ----------
        msg : Any
            Input value for `msg`.

        """
        self.root_logger.set_next_actor_name(self.object_name)
        self.root_logger.msg_typ("INFO", msg)

    def warning(self, msg):
        """Execute warning.

        Parameters
        ----------
        msg : Any
            Input value for `msg`.

        """
        self.root_logger.set_next_actor_name(self.object_name)
        self.root_logger.msg_typ("WARNING", msg)

    def error(self, msg):
        """Execute error.

        Parameters
        ----------
        msg : Any
            Input value for `msg`.

        """
        self.root_logger.set_next_actor_name(self.object_name)
        self.root_logger.msg_typ("ERROR", msg)

    def exception(self, msg):
        """Execute exception.

        Parameters
        ----------
        msg : Any
            Input value for `msg`.

        """
        self.root_logger.set_next_actor_name(self.object_name)
        self.root_logger.msg_typ("EXCEPTION", msg)

    def critical(self, msg):
        """Execute critical.

        Parameters
        ----------
        msg : Any
            Input value for `msg`.

        """
        self.root_logger.set_next_actor_name(self.object_name)
        self.root_logger.msg_typ("CRITICAL", msg)


class FileLogger(SimpleLogger):
    """Represent the filelogger component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

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
        """Initialize object state and default configuration.

        Parameters
        ----------
        filename_log : Any, optional
            Input value for `filename_log`.
        write_to_stdout : Any, optional
            Input value for `write_to_stdout`.
        level : Any, optional
            Input value for `level`.
        encoding : Any, optional
            Input value for `encoding`.

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
        """
        Sets the gui_log handler

        Parameters
        ----------
        handler : CustomConsoleHandler
            A custom console handler

        """
        self.gui_log = handler

    def set_next_actor_name(self, actor_name):
        """
        Sets the name of the object calling the log

        Parameters
        ----------
        actor_name : str
            The name of the object calling the log

        """
        self.next_actor_name = actor_name

    def set_min_console_level(self, new_level):
        """
        Sets the minimum message level the console will display

        Parameters
        ----------
        new_level : str in ["DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION", "CRITICAL"]
            The name of the minimum message level

        """
        assert new_level in self.msg_level_dict
        self.min_console_level = new_level

    def set_min_filelog_level(self, new_level):
        """
        Sets the minimum message level the log will include

        Parameters
        ----------
        new_level : str in ["DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION", "CRITICAL"]
            The name of the minimum message level

        """
        assert new_level in self.msg_level_dict
        self.min_filelog_level = new_level

    def set_filename_log(self, filename_log):
        """Set filename log.

        Parameters
        ----------
        filename_log : Any
            Input value for `filename_log`.

        """
        assert self.open_file is None
        self.filename_log = filename_log
        self.open_file = open_file(self.filename_log)
        self.write_to_file = self.open_file is not None and not self.open_file.closed

    def close_file(self):
        """Execute close file.

        Notes
        -----
        This method performs internal setup or validation logic.

        """
        if self.open_file:
            self.open_file.close()
        self.open_file = None
        self.write_to_file = False

    def log_function_handler(self, typ, filename, lineno, msg):
        """
        Default logging function. Takes a text and outputs to stdout.

        Parameters
        ----------
        typ : str
            message type - ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        filename : str
            the active file
        lineno : int
            line number
        msg : str
            message to be displayed

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
        """
        Default logging function. Takes a text and outputs to stdout.
        Parameters
        ----------
        typ : str
            message type - ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        filename : str
            the active file
        lineno : int
            line number
        msg : str
            message to be displayed
        Message will have format 'typ: msg'
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
        """
        Default logging function. Takes a text and outputs to stdout.

        Parameters
        ----------
        typ : str
            message type - ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        filename : str
            the active file
        lineno : int
            line number
        msg : str
            message to be displayed

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
        """
        Begins capturing all messages above a certain level in an internal record

        Parameters
        ----------
        capture_level_min : str in ["DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION", "CRITICAL"]
            The name of the minimum message level

        clear_capture_log : bool
            If true, clears the previous internal record

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
        """
        Returns the internal captured log

        Returns
        ----------
        capture_log : [(str, str)] [(typ, msg)]
            A list of tuples containing the msg type and the msg
        """
        return self.capture_log

    def end_log_capture(self):
        """
        Ends the capturing of messages and returns the capture log

        Returns
        ----------
        capture_log : [(str, str)] [(typ, msg)]
            A list of tuples containing the msg type and the msg
        """
        self.do_capture_log = False
        return self.capture_log

    def __del__(self):
        """Execute del.

        Notes
        -----
        This method performs internal setup or validation logic.

        """
        if self.open_file is not None and not self.open_file.closed:
            self.open_file.close()

    def __repr__(self):
        """Return a developer-oriented string representation.

        Returns
        -------
        Any
            Return value produced by this operation.

        """
        return (
            "FileLogger(filename_log=%r, write_to_stdout=%r, level=%r, encoding=%r)"
            % (self.filename_log, self.write_to_file, self.level, self.encoding)
        )


def open_file(filename_log):
    """Execute open file.

    Parameters
    ----------
    filename_log : Any
        Input value for `filename_log`.

    Returns
    -------
    Any
        Return value produced by this operation.

    """
    open_file = None
    if filename_log is not None:
        open_file = open(filename_log, "w")
    return open_file
