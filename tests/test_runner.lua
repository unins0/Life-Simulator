-- test_runner.lua — minimal pure-Lua test runner (works with lua AND luajit).
-- Usage (from project root, cwd == root):
--   lua tests/test_runner.lua [path/to/test_file.lua ...]
--   luajit tests/test_runner.lua
-- With no arguments, auto-discovers tests/test_*.lua files.
-- Each test file must return a table: { name = function() ... end }.
-- Exits 0 if all tests pass, 1 otherwise. No love.* dependencies.

local runner = {}

-- We run from the project root (cwd), so the root itself is where modules live.
local ROOT = '.' -- e.g. 'shares.lua', 'ai_module.lua'
package.path = ROOT .. '/?.lua;' .. package.path

-- Load a test file: dofile + pcall, must return a table {name = fn}.
function runner.load_test_file(path)
    local chunk, err = loadfile(path)
    if not chunk then
        return nil, ("cannot load %s: %s"):format(path, err)
    end
    local ok, result = pcall(chunk)
    if not ok then
        return nil, ("error loading %s: %s"):format(path, result)
    end
    if type(result) ~= 'table' then
        return nil, ("%s must return a table of {name = function()} tests"):format(path)
    end
    return result
end

-- Auto-discover tests/test_*.lua files. Uses lfs if available,
-- otherwise falls back to a shell glob (linux/mac).
function runner.discover_tests()
    local files = {}
    local ok, lfs = pcall(require, 'lfs')
    if ok then
        for name in lfs.dir('tests') do
            if name:match('^test_.+%.lua$') and name ~= 'test_runner.lua' then
                files[#files + 1] = 'tests/' .. name
            end
        end
    else
        local p = io.popen("ls tests/test_*.lua 2>/dev/null | grep -v test_runner.lua")
        if p then
            for line in p:lines() do
                files[#files + 1] = line:gsub('%s+$', '')
            end
            p:close()
        end
    end
    table.sort(files)
    return files
end

-- Run a single test file. Returns passed, failed counts.
function runner.run_file(path)
    local tests, err = runner.load_test_file(path)
    if not tests then
        io.write(("  FAIL  %s  (%s)\n"):format(path, err))
        return 0, 1
    end

    local names = {}
    for name in pairs(tests) do names[#names + 1] = name end
    table.sort(names)

    local passed, failed = 0, 0
    io.write(("== %s (%d test(s)) ==\n"):format(path, #names))
    for _, name in ipairs(names) do
        local fn = tests[name]
        if type(fn) ~= 'function' then
            io.write(("  FAIL  %s  (entry is %s, not function)\n"):format(name, type(fn)))
            failed = failed + 1
        else
            local ok, err = pcall(fn)
            if ok then
                io.write(("  PASS  %s\n"):format(name))
                passed = passed + 1
            else
                io.write(("  FAIL  %s  (%s)\n"):format(name, err))
                failed = failed + 1
            end
        end
    end
    return passed, failed
end

function runner.main(args)
    local files = {}
    for _, a in ipairs(args) do
        if a:match('%.lua$') then files[#files + 1] = a end
    end
    if #files == 0 then files = runner.discover_tests() end

    if #files == 0 then
        io.write("No test files given and none auto-discovered.\n")
        os.exit(0)
    end

    local total_passed, total_failed = 0, 0
    for _, path in ipairs(files) do
        local p, f = runner.run_file(path)
        total_passed = total_passed + p
        total_failed = total_failed + f
    end

    io.write(("\n%d/%d passed\n"):format(total_passed, total_passed + total_failed))
    os.exit(total_failed > 0 and 1 or 0)
end

-- arg[1..] = CLI arguments when run as a script.
if arg and arg[0] and arg[0]:match('test_runner%.lua$') then
    runner.main(arg)
end

return runner
