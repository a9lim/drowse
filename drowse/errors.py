"""Error taxonomy for the engine layer.

Every custom exception defined in drowse re-parents to :class:`DrowseError`
so callers (and the HTTP server) can catch the whole family with a single
``except DrowseError``. Stdlib parents (``ValueError``, ``RuntimeError``,
``KeyError``, ``ImportError``, ...) stay in the MRO so generic
``except ValueError`` sites catch the relevant subclasses too.

Every subclass returns an HTTP-style status code through
:meth:`DrowseError.user_message`, which the server and CLI consume to translate
exceptions consistently. The default is ``(500, str(self))``; subclasses
lift the status (and optionally rewrite the message) by overriding the
method.

**Request errors.**  Input a caller can get wrong — a malformed spec, an
out-of-range parameter, a name that cannot apply — raises a
:class:`RequestError` (400, text verbatim) at the point it is validated.  A
*bare* ``ValueError`` means an engine invariant broke: the server logs it
and answers a scrubbed 500, so its text never reaches a client.  When a new
validation can be triggered by a request, raise from this family, not a
bare ``ValueError``.

**Ownership rule.**  This module is the single home for the *core engine's*
error classes — the ones raised across module boundaries by
``manifold``/``mahalanobis``/``profile``/``steering_expr``/``hooks``/``sae``.
Defining them here rather than beside each raiser is what keeps the taxonomy
visible in one place and keeps a low-level module (``mahalanobis``,
``profile``) from becoming an import dependency of everything that catches
its errors.  Modules that historically owned a class re-export it under its
old name so ``from drowse.geometry.whitener import WhitenerError`` and friends
keep working; those aliases are import-compatibility only, not a second home.
The io layer keeps its own errors beside the format code they describe
(``ManifoldFormatError``, ``SelectorError``, ``TemplateNotFoundError``, ...) —
they never cross into core.
"""


class DrowseError(Exception):
    """Base class for all drowse-raised errors.

    Subclasses override :meth:`user_message` to provide an HTTP-style
    status code (``400`` bad input, ``404`` not found, ``409`` conflict,
    ``422`` semantic-but-syntactically-valid, ``500`` server error,
    ``502`` upstream).  The CLI maps the status to an exit code via
    ``min(2, code // 100)``; the HTTP server passes it through.
    """

    def user_message(self) -> tuple[int, str]:
        """Return ``(status_code, formatted_message)`` for user-facing surfaces."""
        return (500, str(self) or self.__class__.__name__)


def is_out_of_memory_error(exc: BaseException) -> bool:
    """Recognize accelerator and CPU allocator OOM spellings."""
    message = str(exc).lower()
    return any(
        needle in message
        for needle in ("out of memory", "can't allocate memory", "cannot allocate memory")
    )


class RequestError(ValueError, DrowseError):
    """Input the caller got wrong: malformed, out of range, or naming
    something that cannot apply.

    Raised where the input is validated, so the server answers 400 with this
    text verbatim — the text is the user-facing contract.  A bare
    ``ValueError`` is an engine invariant instead, which the server logs and
    scrubs to a 500; anything a client can trigger by sending a bad value
    belongs in this family.  Subclasses name a failure a client keys on;
    the rest raise this directly.
    """

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class GenerationRequestError(RequestError):
    """A generation or scoring request the caller got wrong (bad ``n``, an
    empty choice set, mismatched labels): a 400, never an engine failure."""


class SourceSpecError(RequestError):
    """A malformed or unknown source specification: a lens or SAE source
    (``local:<name>``, a provider, a release), a Hugging Face corpus spec,
    or a manifold install target."""


class ManifoldSpecError(RequestError):
    """A manifold authoring or fit request the geometry cannot honor: an
    invalid domain spec, too few nodes for the resolved topology, or fit
    overrides that do not apply to the folder."""


class PoisednessError(ManifoldSpecError):
    """The node coordinates cannot determine an RBF interpolant (too few
    nodes, or nodes that do not affinely span the embedding).  The client
    keys its layout hint on this class name, which is the wire ``code``."""


class WhitenerError(ValueError, DrowseError):
    """Raised when whitener construction or lookup fails.

    Also the all-or-nothing metric gate: every activation-space surface
    (fit, projection, monitor read, cross-model rebake, ``manifold compare``)
    requires a :class:`~drowse.geometry.whitener.LayerWhitener` covering every
    scored layer and raises this otherwise — there is no Euclidean fallback.
    Re-exported from ``geometry.whitener``.
    """

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class ProfileError(ValueError, DrowseError):
    """Raised on invalid Profile operations (missing layer, empty, etc.).

    Re-exported from ``geometry.profile``.
    """

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class MergeError(RequestError):
    """Raised when an offline direction merge or bake cannot be built.

    Covers the additive-subset grammar checks of ``manifold bake`` and the
    layer-coverage check of :func:`~drowse.geometry.profile.linear_sum`.
    """


class UnknownManifoldLabelError(KeyError, DrowseError):
    """Raised when a manifold position payload names an unknown node label.

    Produced by :meth:`~drowse.geometry.manifold.Manifold.resolve_position` (and
    the nearest-node helpers, which short-circuit on labels) when the label is
    not in ``Manifold.node_labels``.  Surfaces a 404-shaped error at the HTTP
    layer through the shared :class:`DrowseError` MRO; CLI handlers print the
    message and recover.  Re-exported from ``geometry.manifold``.
    """

    def user_message(self) -> tuple[int, str]:
        # ``str(KeyError("x"))`` is repr-quoted; ``args[0]`` is the message.
        msg = self.args[0] if self.args else self.__class__.__name__
        return (404, str(msg))


class SaeBackendImportError(ImportError, DrowseError):
    """Raised when sae_lens is required but not installed."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class SaeReleaseNotFoundError(RequestError):
    """Raised when a requested SAELens release does not exist."""


class SaeModelMismatchError(RequestError):
    """Raised when an SAE's base model does not match the drowse model."""


class SaeCoverageError(RequestError):
    """Raised when an SAE release covers zero of the model's layers."""


class SaeNotLoadedError(RuntimeError, DrowseError):
    """Raised when a live SAE surface is used without a resident release."""

    def user_message(self) -> tuple[int, str]:
        return (404, str(self) or self.__class__.__name__)


class SaeFeatureError(RequestError):
    """Raised when an SAE feature id is malformed or outside the resident width."""


class RelpUnsupportedError(RuntimeError, DrowseError):
    """Raised when a model's decoder layers do not fit the dense RelP rule
    shape (one standard gated MLP plus residual RMSNorms per layer) — MoE
    experts and fused gate-up projections need per-arch rule arms that are
    not implemented, and a silently partial RelP fit would be a wrong
    artifact rather than a degraded one.
    """

    def user_message(self) -> tuple[int, str]:
        return (422, str(self) or self.__class__.__name__)


class UnsupportedProbeChannelError(RequestError):
    """Raised at steering-composition preflight when a probe gate references
    a channel its instrument family can never produce (e.g.
    ``@when:sae/123:membership`` — SAE readings carry only the activation
    axis).  Distinct from a *supported* channel that is temporarily absent
    this step (prefill, capture unavailable), which stays quietly inactive.
    """


class AmbiguousVariantError(RequestError):
    """Raised when a :sae selector matches more than one extracted release."""


class UnknownVariantError(KeyError, DrowseError):
    """Raised when a variant selector does not match any on-disk tensor."""

    def user_message(self) -> tuple[int, str]:
        # ``str(KeyError("x"))`` is ``"'x'"`` (repr-quoted); use ``args[0]``
        # when present so the user sees the original message.
        msg = self.args[0] if self.args else self.__class__.__name__
        return (404, str(msg))


class RoleBaselineMismatchWarning(UserWarning):
    """Warns that a role-augmented steering expression mixes a plain term
    in.  The plain term's baseline was the family's standard ``assistant``
    role label; the role-augmented terms substitute a custom role label
    into the chat-template render.  Composing them is supported but the
    plain term's baseline doesn't track the substituted role, so the
    interaction may behave unexpectedly.  The warning fires once per
    mixed-baseline ``steering()`` scope.
    """


class SteeringExprError(RequestError):
    """Raised when a steering expression string cannot be parsed."""

    def __init__(self, msg: str, *, col: int | None = None) -> None:
        self.col = col
        if col is not None:
            msg = f"{msg} (col {col})"
        super().__init__(msg)


class SteeringCompositionError(ValueError, DrowseError):
    """Base for failures that happen *after* an expression parses cleanly.

    The expression was syntactically valid and every atom resolved; what
    failed is composing the resolved terms against the loaded artifacts —
    a coordinate count that doesn't match the manifold's domain, or two
    curved manifolds whose spans collide at a shared layer, or a negative
    coefficient on a curved term.  Raised by ``SteeringPlan.add_manifold`` /
    ``SteeringPlan.lower`` and the affine push, never by the parser.

    Status is ``422``: syntactically valid, semantically unsatisfiable.
    That is the distinction from :class:`SteeringExprError`'s ``400`` — a
    parse failure is malformed input, a composition failure is a
    well-formed request the loaded geometry cannot honor.
    """

    def user_message(self) -> tuple[int, str]:
        return (422, str(self) or self.__class__.__name__)


class ManifoldArityError(SteeringCompositionError):
    """Raised when a ``%`` manifold position has the wrong number of
    coordinates for the manifold's domain.

    The grammar collects the position payload but cannot validate arity —
    it does not know the domain.  ``SteeringPlan.add_manifold`` checks
    the coordinate count against the loaded domain's intrinsic dimension
    and raises this when they disagree.
    """


class OverlappingManifoldError(SteeringCompositionError):
    """Raised when curved manifold terms overlap at a shared layer.

    Multiple curved terms may share a layer when their fitted spans are
    near-orthogonal; the affine merged subspace is orthogonalized against
    those spans.  This error identifies the unsupported case where two curved
    spans exceed the overlap tolerance.
    """


class ManifoldNotFoundError(FileNotFoundError, DrowseError):
    """Raised when a manifold folder or its fitted tensor is not found.

    Preserves ``FileNotFoundError`` in the MRO so existing
    ``except FileNotFoundError`` call sites (server, CLI) still catch it.
    """

    def user_message(self) -> tuple[int, str]:
        return (404, str(self) or self.__class__.__name__)


class ManifoldExistsError(FileExistsError, DrowseError):
    """Raised when a manifold tensor already exists and ``force`` is off.

    Preserves ``FileExistsError`` in the MRO so existing
    ``except FileExistsError`` call sites (server, CLI) still catch it.
    """

    def user_message(self) -> tuple[int, str]:
        return (409, str(self) or self.__class__.__name__)


class ConcurrentGenerationError(RuntimeError, DrowseError):
    """Raised when a generation call is made while another is in progress."""

    def user_message(self) -> tuple[int, str]:
        return (409, str(self) or self.__class__.__name__)


class ProfileNotRegisteredError(KeyError, DrowseError):
    """Raised when a steering call references a profile not in the registry."""

    def user_message(self) -> tuple[int, str]:
        # KeyError str-formats the message as repr; reach into args
        # so the user sees the original text.
        msg = self.args[0] if self.args else self.__class__.__name__
        return (404, str(msg))


class ManifoldNotRegisteredError(KeyError, DrowseError):
    """Raised when manifold steering references an unknown / unfitted manifold."""

    def user_message(self) -> tuple[int, str]:
        msg = self.args[0] if self.args else self.__class__.__name__
        return (404, str(msg))


class ConcurrentExtractionError(RuntimeError, DrowseError):
    """Raised when ``session.extract`` is called while a generation is in flight.

    Mirrors :class:`ConcurrentGenerationError` — extraction runs forward
    passes through the model and would race with an active generation if
    allowed to overlap.  Also raised for artifact-producing work (neutral
    baseline, fits) requested inside an active steering scope.
    """

    def user_message(self) -> tuple[int, str]:
        return (409, str(self) or self.__class__.__name__)
