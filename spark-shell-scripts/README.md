# Spark-shell scripts 

The scripts in this directory are intended to run on a spark-shell session.

## Shell tips and tricks

__Loading custom libraries in the shell__
 
When starting a spark shell, you can specify any number of jar files using the `--jars` option. For example:

    spark-shell --jars target/scala-2.11/bda-project-lsa-assembly-1.0.jar


__Executing a script in the shell__

You can run a scala script inside the shell using the `:load` directive. The filepath is relative to the pwd where you launched the shell. Say you ran the shell at the root of the project, you can use:

    :load spark-shell-scripts/load-data-example.scala

This will create the tf-idf matrix and the document and term lookups. Note that the script runs inside the shell, so any variable previously defined is available to it.

__Spark-shell commands__

Special commands are prefixed by ":". Use `:help` to display a list:

```
scala> :help
All commands can be abbreviated, e.g., :he instead of :help.
:edit <id>|<line>        edit history
:help [command]          print this summary or command-specific help
:history [num]           show the history (optional num is commands to show)
:h? <string>             search the history
:imports [name name ...] show import history, identifying sources of names
:implicits [-v]          show the implicits in scope
:javap <path|class>      disassemble a file or class name
:line <id>|<line>        place line(s) at the end of history
:load <path>             interpret lines in a file
:paste [-raw] [path]     enter paste mode or paste a file
:power                   enable power user mode
:quit                    exit the interpreter
:replay [options]        reset the repl and replay all previous commands
:require <path>          add a jar to the classpath
:reset [options]         reset the repl to its initial state, forgetting all session entries
:save <path>             save replayable session to a file
:sh <command line>       run a shell command (result is implicitly => List[String])
:settings <options>      update compiler options, if possible; see reset
:silent                  disable/enable automatic printing of results
:type [-v] <expr>        display the type of an expression without evaluating it
:kind [-v] <expr>        display the kind of expression's type
:warnings                show the suppressed warnings from the most recent line which had any
 
```

__Spark log__

By default, the log level is `INFO`. You can modify it from the shell using:

    sc.setLogLevel("WARN")
    
See also the `:silent` command.
