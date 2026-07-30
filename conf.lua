function love.conf(t)
    t.version = '11.5'
    t.console = true

    t.window.title     = 'Life Simulator v5.0'
    t.window.width     = 800
    t.window.height    = 600
    t.window.resizable = true
    t.window.msaa      = 0

    t.modules.video    = false
    t.modules.touch    = false
    t.modules.physics  = false
    t.modules.joystick = false
end