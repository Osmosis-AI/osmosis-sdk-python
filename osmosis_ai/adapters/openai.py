"""
OpenAI adapter for Osmosis

This module provides monkey patching for the OpenAI Python client.
"""

import functools
import inspect
import sys

from osmosis_ai import utils
from osmosis_ai.utils import send_to_osmosis
from osmosis_ai.logger import logger


def wrap_openai() -> None:
    """
    Monkey patch OpenAI's client to send all prompts and responses to OSMOSIS.

    This function should be called before creating any OpenAI client instances.
    """
    try:
        import openai
    except ImportError:
        logger.debug("openai package is not installed.")
        return

    # Try to detect which version of the OpenAI client is installed
    try:
        from openai import OpenAI

        # Check for v2 client first
        try:
            import openai.version

            if openai.version.__version__.startswith("2."):
                _ai_openai_v2()
                return
        except (ImportError, AttributeError):
            pass

        # Fall back to v1 client
        _ai_openai_v1()
    except (ImportError, AttributeError):
        # Fall back to legacy client
        _ai_openai_legacy()


def _ai_openai_v2() -> None:
    """Monkey patch the OpenAI v2 client."""
    import openai

    try:
        # Get the OpenAI class
        from openai import OpenAI

        # Debug: Print all available OpenAI modules
        logger.debug(f"OpenAI modules: {dir(openai)}")

        # Try to import AsyncOpenAI
        try:
            from openai import AsyncOpenAI

            logger.info(f"Successfully imported AsyncOpenAI")
            has_async_client = True
        except ImportError:
            logger.warning(f"Failed to import AsyncOpenAI")
            has_async_client = False

        # Store the original __init__ method for OpenAI
        original_init = OpenAI.__init__

        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            # Call the original __init__
            original_init(self, *args, **kwargs)

            # Debug: Print client structure
            if hasattr(self, "chat") and hasattr(self.chat, "completions"):
                logger.debug(
                    f"Client chat completions methods: {dir(self.chat.completions)}"
                )
                if hasattr(self.chat.completions, "create"):
                    logger.debug(
                        f"create is a coro: {inspect.iscoroutinefunction(self.chat.completions.create)}"
                    )

            # Now wrap the client's chat.completions.create and completions.create methods
            if hasattr(self, "chat") and hasattr(self.chat, "completions"):
                original_chat_create = self.chat.completions.create
                if not hasattr(original_chat_create, "_osmosis_aiped"):

                    @functools.wraps(original_chat_create)
                    def wrapped_chat_create(*args, **kwargs):
                        # Check if streaming is enabled
                        is_streaming = kwargs.get("stream", False)

                        if is_streaming:
                            # For streaming, we need to wrap the iterator
                            stream = original_chat_create(*args, **kwargs)

                            if utils.enabled:
                                # Create a new wrapped iterator that sends data to OSMOSIS
                                def wrapped_stream():
                                    chunks = []
                                    for chunk in stream:
                                        chunks.append(
                                            chunk.model_dump()
                                            if hasattr(chunk, "model_dump")
                                            else chunk
                                        )
                                        yield chunk

                                    # After collecting all chunks, send them to OSMOSIS
                                    if utils.enabled:
                                        send_to_osmosis(
                                            query=kwargs,
                                            response={"streaming_chunks": chunks},
                                            status=200,
                                        )

                                return wrapped_stream()
                            else:
                                return stream
                        else:
                            # For non-streaming, handle normally
                            response = original_chat_create(*args, **kwargs)

                            if utils.enabled:
                                send_to_osmosis(
                                    query=kwargs,
                                    response=(
                                        response.model_dump()
                                        if hasattr(response, "model_dump")
                                        else response
                                    ),
                                    status=200,
                                )

                            return response

                    wrapped_chat_create._osmosis_aiped = True
                    self.chat.completions.create = wrapped_chat_create

            if hasattr(self, "completions"):
                original_completions_create = self.completions.create
                if not hasattr(original_completions_create, "_osmosis_aiped"):

                    @functools.wraps(original_completions_create)
                    def wrapped_completions_create(*args, **kwargs):
                        # Check if streaming is enabled
                        is_streaming = kwargs.get("stream", False)

                        if is_streaming:
                            # For streaming, we need to wrap the iterator
                            stream = original_completions_create(*args, **kwargs)

                            if utils.enabled:
                                # Create a new wrapped iterator that sends data to OSMOSIS
                                def wrapped_stream():
                                    chunks = []
                                    for chunk in stream:
                                        chunks.append(
                                            chunk.model_dump()
                                            if hasattr(chunk, "model_dump")
                                            else chunk
                                        )
                                        yield chunk

                                    # After collecting all chunks, send them to OSMOSIS
                                    if utils.enabled:
                                        send_to_osmosis(
                                            query=kwargs,
                                            response={"streaming_chunks": chunks},
                                            status=200,
                                        )

                                return wrapped_stream()
                            else:
                                return stream
                        else:
                            # For non-streaming, handle normally
                            response = original_completions_create(*args, **kwargs)

                            if utils.enabled:
                                send_to_osmosis(
                                    query=kwargs,
                                    response=(
                                        response.model_dump()
                                        if hasattr(response, "model_dump")
                                        else response
                                    ),
                                    status=200,
                                )

                            return response

                    wrapped_completions_create._osmosis_aiped = True
                    self.completions.create = wrapped_completions_create

            # Wrap async methods
            if hasattr(self, "chat") and hasattr(self.chat, "completions"):
                if hasattr(self.chat.completions, "acreate"):
                    logger.debug(f"Found acreate in chat.completions")
                    original_achat_create = self.chat.completions.acreate
                    if not hasattr(original_achat_create, "_osmosis_aiped"):

                        @functools.wraps(original_achat_create)
                        async def wrapped_achat_create(*args, **kwargs):
                            # Check if streaming is enabled
                            is_streaming = kwargs.get("stream", False)

                            if is_streaming:
                                # For streaming, we need to wrap the async iterator
                                stream = await original_achat_create(*args, **kwargs)

                                if utils.enabled:
                                    # Create a new wrapped async iterator that sends data to OSMOSIS
                                    async def wrapped_stream():
                                        chunks = []
                                        async for chunk in stream:
                                            chunks.append(
                                                chunk.model_dump()
                                                if hasattr(chunk, "model_dump")
                                                else chunk
                                            )
                                            yield chunk

                                        # After collecting all chunks, send them to OSMOSIS
                                        if utils.enabled:
                                            send_to_osmosis(
                                                query=kwargs,
                                                response={"streaming_chunks": chunks},
                                                status=200,
                                            )

                                    return wrapped_stream()
                                else:
                                    return stream
                            else:
                                # For non-streaming, handle normally
                                response = await original_achat_create(*args, **kwargs)

                                if utils.enabled:
                                    send_to_osmosis(
                                        query=kwargs,
                                        response=(
                                            response.model_dump()
                                            if hasattr(response, "model_dump")
                                            else response
                                        ),
                                        status=200,
                                    )

                                return response

                        wrapped_achat_create._osmosis_aiped = True
                        self.chat.completions.acreate = wrapped_achat_create
                else:
                    logger.debug(f"acreate not found in chat.completions")

            if hasattr(self, "completions"):
                if hasattr(self.completions, "acreate"):
                    original_acompletions_create = self.completions.acreate
                    if not hasattr(original_acompletions_create, "_osmosis_aiped"):

                        @functools.wraps(original_acompletions_create)
                        async def wrapped_acompletions_create(*args, **kwargs):
                            # Check if streaming is enabled
                            is_streaming = kwargs.get("stream", False)

                            if is_streaming:
                                # For streaming, we need to wrap the async iterator
                                stream = await original_acompletions_create(
                                    *args, **kwargs
                                )

                                if utils.enabled:
                                    # Create a new wrapped async iterator that sends data to OSMOSIS
                                    async def wrapped_stream():
                                        chunks = []
                                        async for chunk in stream:
                                            chunks.append(
                                                chunk.model_dump()
                                                if hasattr(chunk, "model_dump")
                                                else chunk
                                            )
                                            yield chunk

                                        # After collecting all chunks, send them to OSMOSIS
                                        if utils.enabled:
                                            send_to_osmosis(
                                                query=kwargs,
                                                response={"streaming_chunks": chunks},
                                                status=200,
                                            )

                                    return wrapped_stream()
                                else:
                                    return stream
                            else:
                                # For non-streaming, handle normally
                                response = await original_acompletions_create(
                                    *args, **kwargs
                                )

                                if utils.enabled:
                                    send_to_osmosis(
                                        query=kwargs,
                                        response=(
                                            response.model_dump()
                                            if hasattr(response, "model_dump")
                                            else response
                                        ),
                                        status=200,
                                    )

                                return response

                        wrapped_acompletions_create._osmosis_aiped = True
                        self.completions.acreate = wrapped_acompletions_create

        wrapped_init._osmosis_aiped = True
        OpenAI.__init__ = wrapped_init

        # Also wrap AsyncOpenAI if it exists
        if has_async_client:
            logger.debug(f"Wrapping AsyncOpenAI __init__")
            original_async_init = AsyncOpenAI.__init__

            @functools.wraps(original_async_init)
            def wrapped_async_init(self, *args, **kwargs):
                # Call the original __init__
                original_async_init(self, *args, **kwargs)

                # Debug: Print AsyncOpenAI client structure
                logger.debug(f"AsyncOpenAI client structure:")
                if hasattr(self, "chat") and hasattr(self.chat, "completions"):
                    logger.debug(
                        f"AsyncOpenAI chat completions methods: {dir(self.chat.completions)}"
                    )
                    if hasattr(self.chat.completions, "create"):
                        logger.debug(
                            f"create is a coro: {inspect.iscoroutinefunction(self.chat.completions.create)}"
                        )

                # Wrap the async client's methods
                if hasattr(self, "chat") and hasattr(self.chat, "completions"):
                    original_achat_create = self.chat.completions.create
                    if not hasattr(original_achat_create, "_osmosis_aiped"):
                        logger.debug(f"Wrapping AsyncOpenAI chat.completions.create")

                        @functools.wraps(original_achat_create)
                        async def wrapped_achat_create(*args, **kwargs):
                            logger.debug(f"AsyncOpenAI wrapped create called")
                            response = await original_achat_create(*args, **kwargs)

                            if utils.enabled:
                                send_to_osmosis(
                                    query=kwargs,
                                    response=(
                                        response.model_dump()
                                        if hasattr(response, "model_dump")
                                        else response
                                    ),
                                    status=200,
                                )

                            return response

                        wrapped_achat_create._osmosis_aiped = True
                        self.chat.completions.create = wrapped_achat_create

                if hasattr(self, "completions"):
                    original_acompletions_create = self.completions.create
                    if not hasattr(original_acompletions_create, "_osmosis_aiped"):

                        @functools.wraps(original_acompletions_create)
                        async def wrapped_acompletions_create(*args, **kwargs):
                            response = await original_acompletions_create(
                                *args, **kwargs
                            )

                            if utils.enabled:
                                send_to_osmosis(
                                    query=kwargs,
                                    response=(
                                        response.model_dump()
                                        if hasattr(response, "model_dump")
                                        else response
                                    ),
                                    status=200,
                                )

                            return response

                        wrapped_acompletions_create._osmosis_aiped = True
                        self.completions.create = wrapped_acompletions_create

            wrapped_async_init._osmosis_aiped = True
            AsyncOpenAI.__init__ = wrapped_async_init

        logger.info("OpenAI v2 client has been wrapped by osmosis-ai.")
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to wrap OpenAI v2 client: {e}")


def _ai_openai_v1() -> None:
    """Monkey patch the OpenAI v1 client."""
    from openai import OpenAI
    from openai.resources.chat import completions
    from openai.resources import completions as text_completions

    # Print package structure to debug
    logger.debug(f"OpenAI package structure in v1 wrapper:")
    try:
        import openai

        logger.debug(f"OpenAI modules: {dir(openai)}")
    except Exception as e:
        logger.error(f"Error inspecting openai module: {e}")

    # Try to import AsyncOpenAI
    try:
        from openai import AsyncOpenAI

        logger.info(f"Successfully imported AsyncOpenAI in v1 wrapper")
        has_async_client = True
    except ImportError:
        logger.warning(f"Failed to import AsyncOpenAI in v1 wrapper")
        has_async_client = False

    # Print available methods in completions module
    logger.debug(f"Available methods in completions: {dir(completions.Completions)}")

    # Patch the chat completions create method
    original_chat_create = completions.Completions.create
    if not hasattr(original_chat_create, "_osmosis_aiped"):

        @functools.wraps(original_chat_create)
        def wrapped_chat_create(self, *args, **kwargs):
            # Check if streaming is enabled
            is_streaming = kwargs.get("stream", False)

            if is_streaming:
                # For streaming, we need to wrap the iterator
                stream = original_chat_create(self, *args, **kwargs)

                if utils.enabled:
                    # Create a new wrapped iterator that sends data to OSMOSIS
                    def wrapped_stream():
                        chunks = []
                        for chunk in stream:
                            chunks.append(
                                chunk.model_dump()
                                if hasattr(chunk, "model_dump")
                                else chunk
                            )
                            yield chunk

                        # After collecting all chunks, send them to OSMOSIS
                        if utils.enabled:
                            send_to_osmosis(
                                query=kwargs,
                                response={"streaming_chunks": chunks},
                                status=200,
                            )

                    return wrapped_stream()
                else:
                    return stream
            else:
                # For non-streaming, handle normally
                response = original_chat_create(self, *args, **kwargs)

                if utils.enabled:
                    send_to_osmosis(
                        query=kwargs,
                        response=(
                            response.model_dump()
                            if hasattr(response, "model_dump")
                            else response
                        ),
                        status=200,
                    )

                return response

        wrapped_chat_create._osmosis_aiped = True
        completions.Completions.create = wrapped_chat_create

    # Patch the completions create method
    original_completions_create = text_completions.Completions.create
    if not hasattr(original_completions_create, "_osmosis_aiped"):

        @functools.wraps(original_completions_create)
        def wrapped_completions_create(self, *args, **kwargs):
            # Check if streaming is enabled
            is_streaming = kwargs.get("stream", False)

            if is_streaming:
                # For streaming, we need to wrap the iterator
                stream = original_completions_create(self, *args, **kwargs)

                if utils.enabled:
                    # Create a new wrapped iterator that sends data to OSMOSIS
                    def wrapped_stream():
                        chunks = []
                        for chunk in stream:
                            chunks.append(
                                chunk.model_dump()
                                if hasattr(chunk, "model_dump")
                                else chunk
                            )
                            yield chunk

                        # After collecting all chunks, send them to OSMOSIS
                        if utils.enabled:
                            send_to_osmosis(
                                query=kwargs,
                                response={"streaming_chunks": chunks},
                                status=200,
                            )

                    return wrapped_stream()
                else:
                    return stream
            else:
                # For non-streaming, handle normally
                response = original_completions_create(self, *args, **kwargs)

                if utils.enabled:
                    send_to_osmosis(
                        query=kwargs,
                        response=(
                            response.model_dump()
                            if hasattr(response, "model_dump")
                            else response
                        ),
                        status=200,
                    )

                return response

        wrapped_completions_create._osmosis_aiped = True
        text_completions.Completions.create = wrapped_completions_create

    # Find and wrap async methods
    for module in [completions, text_completions]:
        for name, method in inspect.getmembers(module.Completions):
            if (
                name.startswith("a")
                and name.endswith("create")
                and inspect.iscoroutinefunction(method)
                and not hasattr(method, "_osmosis_aiped")
            ):

                logger.debug(f"Found async method {name} in {module.__name__}")
                original_method = method

                @functools.wraps(original_method)
                async def wrapped_async_method(self, *args, **kwargs):
                    # Check if streaming is enabled
                    is_streaming = kwargs.get("stream", False)

                    if is_streaming:
                        # For streaming, we need to wrap the async iterator
                        stream = await original_method(self, *args, **kwargs)

                        if utils.enabled:
                            # Create a new wrapped async iterator that sends data to OSMOSIS
                            async def wrapped_stream():
                                chunks = []
                                async for chunk in stream:
                                    chunks.append(
                                        chunk.model_dump()
                                        if hasattr(chunk, "model_dump")
                                        else chunk
                                    )
                                    yield chunk

                                # After collecting all chunks, send them to OSMOSIS
                                if utils.enabled:
                                    send_to_osmosis(
                                        query=kwargs,
                                        response={"streaming_chunks": chunks},
                                        status=200,
                                    )

                            return wrapped_stream()
                        else:
                            return stream
                    else:
                        # For non-streaming, handle normally
                        response = await original_method(self, *args, **kwargs)

                        if utils.enabled:
                            send_to_osmosis(
                                query=kwargs,
                                response=(
                                    response.model_dump()
                                    if hasattr(response, "model_dump")
                                    else response
                                ),
                                status=200,
                            )

                        return response

                wrapped_async_method._osmosis_aiped = True
                setattr(module.Completions, name, wrapped_async_method)

    # Explicitly wrap AsyncOpenAI if it exists
    if has_async_client:
        logger.debug(f"Wrapping AsyncOpenAI __init__ in v1 wrapper")
        original_async_init = AsyncOpenAI.__init__

        @functools.wraps(original_async_init)
        def wrapped_async_init(self, *args, **kwargs):
            # Call the original __init__
            original_async_init(self, *args, **kwargs)

            # Debug: Print AsyncOpenAI client structure
            logger.debug(f"AsyncOpenAI client structure in v1:")
            if hasattr(self, "chat") and hasattr(self.chat, "completions"):
                logger.debug(
                    f"AsyncOpenAI chat completions methods: {dir(self.chat.completions)}"
                )
                if hasattr(self.chat.completions, "create"):
                    logger.debug(
                        f"create is a coro: {inspect.iscoroutinefunction(self.chat.completions.create)}"
                    )

            # Now wrap the async client's methods
            if hasattr(self, "chat") and hasattr(self.chat, "completions"):
                original_achat_create = self.chat.completions.create
                if not hasattr(original_achat_create, "_osmosis_aiped"):
                    logger.debug(f"Wrapping AsyncOpenAI chat.completions.create in v1")

                    @functools.wraps(original_achat_create)
                    async def wrapped_achat_create(*args, **kwargs):
                        logger.debug(f"AsyncOpenAI v1 wrapped create called")

                        # Check if streaming is enabled
                        is_streaming = kwargs.get("stream", False)

                        if is_streaming:
                            # For streaming, we need to wrap the async iterator
                            stream = await original_achat_create(*args, **kwargs)

                            if utils.enabled:
                                # Create a new wrapped async iterator that sends data to OSMOSIS
                                async def wrapped_stream():
                                    chunks = []
                                    async for chunk in stream:
                                        chunks.append(
                                            chunk.model_dump()
                                            if hasattr(chunk, "model_dump")
                                            else chunk
                                        )
                                        yield chunk

                                    # After collecting all chunks, send them to OSMOSIS
                                    if utils.enabled:
                                        send_to_osmosis(
                                            query=kwargs,
                                            response={"streaming_chunks": chunks},
                                            status=200,
                                        )

                                return wrapped_stream()
                            else:
                                return stream
                        else:
                            # For non-streaming, handle normally
                            response = await original_achat_create(*args, **kwargs)

                            if utils.enabled:
                                send_to_osmosis(
                                    query=kwargs,
                                    response=(
                                        response.model_dump()
                                        if hasattr(response, "model_dump")
                                        else response
                                    ),
                                    status=200,
                                )

                            return response

                    wrapped_achat_create._osmosis_aiped = True
                    self.chat.completions.create = wrapped_achat_create

            if hasattr(self, "completions"):
                original_acompletions_create = self.completions.create
                if not hasattr(original_acompletions_create, "_osmosis_aiped"):

                    @functools.wraps(original_acompletions_create)
                    async def wrapped_acompletions_create(*args, **kwargs):
                        # Check if streaming is enabled
                        is_streaming = kwargs.get("stream", False)

                        if is_streaming:
                            # For streaming, we need to wrap the async iterator
                            stream = await original_acompletions_create(*args, **kwargs)

                            if utils.enabled:
                                # Create a new wrapped async iterator that sends data to OSMOSIS
                                async def wrapped_stream():
                                    chunks = []
                                    async for chunk in stream:
                                        chunks.append(
                                            chunk.model_dump()
                                            if hasattr(chunk, "model_dump")
                                            else chunk
                                        )
                                        yield chunk

                                    # After collecting all chunks, send them to OSMOSIS
                                    if utils.enabled:
                                        send_to_osmosis(
                                            query=kwargs,
                                            response={"streaming_chunks": chunks},
                                            status=200,
                                        )

                                return wrapped_stream()
                            else:
                                return stream
                        else:
                            # For non-streaming, handle normally
                            response = await original_acompletions_create(
                                *args, **kwargs
                            )

                            if utils.enabled:
                                send_to_osmosis(
                                    query=kwargs,
                                    response=(
                                        response.model_dump()
                                        if hasattr(response, "model_dump")
                                        else response
                                    ),
                                    status=200,
                                )

                            return response

                    wrapped_acompletions_create._osmosis_aiped = True
                    self.completions.create = wrapped_acompletions_create

        wrapped_async_init._osmosis_aiped = True
        AsyncOpenAI.__init__ = wrapped_async_init

    logger.info("OpenAI v1 client has been wrapped by osmosis-ai.")


def _ai_openai_legacy() -> None:
    """Monkey patch the legacy OpenAI client."""
    import openai

    # Patch the Completion.create method
    original_completion_create = openai.Completion.create
    if not hasattr(original_completion_create, "_osmosis_aiped"):

        @functools.wraps(original_completion_create)
        def wrapped_completion_create(*args, **kwargs):
            # Check if streaming is enabled
            is_streaming = kwargs.get("stream", False)

            if is_streaming:
                # For streaming, we need to wrap the iterator
                stream = original_completion_create(*args, **kwargs)

                if utils.enabled:
                    # Create a new wrapped iterator that sends data to OSMOSIS
                    def wrapped_stream():
                        chunks = []
                        for chunk in stream:
                            chunks.append(chunk)
                            yield chunk

                        # After collecting all chunks, send them to OSMOSIS
                        if utils.enabled:
                            send_to_osmosis(
                                query=kwargs,
                                response={"streaming_chunks": chunks},
                                status=200,
                            )

                    return wrapped_stream()
                else:
                    return stream
            else:
                # For non-streaming, handle normally
                response = original_completion_create(*args, **kwargs)

                if utils.enabled:
                    send_to_osmosis(query=kwargs, response=response, status=200)

                return response

        wrapped_completion_create._osmosis_aiped = True
        openai.Completion.create = wrapped_completion_create

    # Patch the ChatCompletion.create method
    if hasattr(openai, "ChatCompletion"):
        original_chat_create = openai.ChatCompletion.create
        if not hasattr(original_chat_create, "_osmosis_aiped"):

            @functools.wraps(original_chat_create)
            def wrapped_chat_create(*args, **kwargs):
                # Check if streaming is enabled
                is_streaming = kwargs.get("stream", False)

                if is_streaming:
                    # For streaming, we need to wrap the iterator
                    stream = original_chat_create(*args, **kwargs)

                    if utils.enabled:
                        # Create a new wrapped iterator that sends data to OSMOSIS
                        def wrapped_stream():
                            chunks = []
                            for chunk in stream:
                                chunks.append(chunk)
                                yield chunk

                            # After collecting all chunks, send them to OSMOSIS
                            if utils.enabled:
                                send_to_osmosis(
                                    query=kwargs,
                                    response={"streaming_chunks": chunks},
                                    status=200,
                                )

                        return wrapped_stream()
                    else:
                        return stream
                else:
                    # For non-streaming, handle normally
                    response = original_chat_create(*args, **kwargs)

                    if utils.enabled:
                        send_to_osmosis(query=kwargs, response=response, status=200)

                    return response

            wrapped_chat_create._osmosis_aiped = True
            openai.ChatCompletion.create = wrapped_chat_create

    # Patch the async methods if they exist
    for obj in [openai.Completion, getattr(openai, "ChatCompletion", None)]:
        if obj is None:
            continue

        if hasattr(obj, "acreate"):
            original_acreate = obj.acreate
            if not hasattr(original_acreate, "_osmosis_aiped"):

                @functools.wraps(original_acreate)
                async def wrapped_acreate(*args, **kwargs):
                    # Check if streaming is enabled
                    is_streaming = kwargs.get("stream", False)

                    if is_streaming:
                        # For streaming, we need to wrap the async iterator
                        stream = await original_acreate(*args, **kwargs)

                        if utils.enabled:
                            # Create a new wrapped async iterator that sends data to OSMOSIS
                            async def wrapped_stream():
                                chunks = []
                                async for chunk in stream:
                                    chunks.append(chunk)
                                    yield chunk

                                # After collecting all chunks, send them to OSMOSIS
                                if utils.enabled:
                                    send_to_osmosis(
                                        query=kwargs,
                                        response={"streaming_chunks": chunks},
                                        status=200,
                                    )

                            return wrapped_stream()
                        else:
                            return stream
                    else:
                        # For non-streaming, handle normally
                        response = await original_acreate(*args, **kwargs)

                        if utils.enabled:
                            send_to_osmosis(query=kwargs, response=response, status=200)

                        return response

                wrapped_acreate._osmosis_aiped = True
                obj.acreate = wrapped_acreate

    logger.info("OpenAI legacy client has been wrapped by osmosis-ai.")
