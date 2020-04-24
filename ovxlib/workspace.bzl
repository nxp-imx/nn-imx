def _get_env_var(ctx, name):
    if name in ctx.os.environ:
        return ctx.os.environ[name]
    else:
        return None

def _get_viv_sdk(ctx):
    viv_sdk = _get_env_var(ctx, ctx.attr.local_driver_path_tag)
    return viv_sdk.split(":")[0] if viv_sdk else None

def _safe_symlink(ctx, source, target):
    if ctx.path(source).exists:
        ctx.symlink(source, target)
        return True
    return False

def _info(ctx, message):
    ctx.execute(["echo", "INFO: {}".format(message)], quiet=False)

def _error(ctx, message):
    ctx.execute(["echo", "ERROR: {}".format(message)], quiet=False)

def _driver_archive_impl(ctx):
    use_local_driver = False

    viv_sdk_path = _get_viv_sdk(ctx)
    if viv_sdk_path:
        viv_sdk_path = ctx.path(viv_sdk_path)
        use_local_driver = viv_sdk_path and viv_sdk_path.exists
        if len(str(viv_sdk_path)) > 0 and not use_local_driver:
            _error(ctx, "=" * 50)
            _error(ctx, "{} must point to an existent path".format(ctx.attr.local_driver_path_tag))
            _error(ctx, "=" * 50)
            fail("{} must point to an existent path".format(ctx.attr.local_driver_path_tag))

    if use_local_driver:
        _info(ctx, "=" * 50)
        _info(ctx, "Use driver from local:")
        _info(ctx, "        '{}'".format(viv_sdk_path))
        _info(ctx, "=" * 50)

        # entry: critical
        entries = {
            "bin": False,
            "cfg": False,
            "include": True,
            "lib": False,
            "drivers": False,
            "nnvxc_kernels": False,
            "README": False,
            "vcompiler_cfg": False,
        }
        for entry in entries:
            if not _safe_symlink(ctx, "{}/{}".format(viv_sdk_path, entry), entry) and entries[entry]:
                fail("'{}' cannot be found in VIV_SDK_DIR:{}".format(entry, viv_sdk_path))
    else:
        if not ctx.attr.url and not ctx.attr.urls:
            fail("At least one of url and urls must be provided")

        all_urls = []
        if ctx.attr.urls:
            all_urls = ctx.attr.urls
        if ctx.attr.url:
            all_urls = [ctx.attr.url] + all_urls

        _info(ctx, "=" * 50)
        _info(ctx, "Use driver from http:")
        _info(ctx, "         {}".format(all_urls))
        _info(ctx, "=" * 50)

        ctx.download_and_extract(
            all_urls,
            "",
            ctx.attr.sha256,
            ctx.attr.type,
            ctx.attr.strip_prefix,
        )

    if ctx.attr.build_file != None:
        ctx.template("BUILD.bazel", ctx.attr.build_file, {
            "%prefix%": "external",
        }, False)

driver_archive = repository_rule(
    implementation = _driver_archive_impl,
    attrs = {
        "url": attr.string(),
        "urls": attr.string_list(),
        "sha256": attr.string(),
        "build_file": attr.label(allow_single_file=True),
        "type": attr.string(),
        "strip_prefix": attr.string(),
        "local_driver_path_tag": attr.string(mandatory=True),
    },
)
