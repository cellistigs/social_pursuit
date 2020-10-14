
Script documentation for file: test_joblib_cache, Updated on:2020-10-12 23:49:51.653929
=======================================================================================
 
  
**parent file: [summary_week_10_9_20](./summary_week_10_9_20.md)**
# Summary


This is a training script to see how best to use the joblib cache. We're going to define a function test_redos that is cached, and takes the markdown file itself and a data object as input. This will not activate the cache, but rerun every time in its raw form:  
`test_redos(md,data,ident)`  
However, replacing the markdown document will not trigger a rerun.  
`test_redos(0,data,ident)`

What happens if we change a field of the data object?  
`test_redos(0,data1,ident)`

What happens if we set the ignore flag on the markdown document? It turns out that this works.

The only thing we have to be careful of is to make sure that all cached functions do not themselves add material to the md document, whether that material consists of text or images. 