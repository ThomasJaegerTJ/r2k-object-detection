  *L7?A`?X@B`??"O@)      0=2?
]Iterator::Model::MaxIntraOpParallelism::FiniteSkip::Prefetch::BatchV2::Shuffle::ParallelMapV2??TN{J??!?8_??iB@)??TN{J??1?8_??iB@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::FiniteSkip::Prefetch::BatchV2?;???!Xl&DQ@)i?^`V(??1?m????<@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::FiniteSkip::Prefetch::BatchV2::Shuffle::ParallelMapV2::ParallelMapV2::FlatMap[2]::TFRecordyx??e??!???҇0@)yx??e??1???҇0@:Advanced file read2?
lIterator::Model::MaxIntraOpParallelism::FiniteSkip::Prefetch::BatchV2::Shuffle::ParallelMapV2::ParallelMapV2іs)?*??!K3?A?@)іs)?*??1K3?A?@:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::FiniteSkip::Prefetch::BatchV2::Shuffle?l?????!?u?RD@)???;۔?1o???#&@:Preprocessing2?
uIterator::Model::MaxIntraOpParallelism::FiniteSkip::Prefetch::BatchV2::Shuffle::ParallelMapV2::ParallelMapV2::FlatMap??4?R??!?????3@)???{h??1"ٰVq@:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteSkip?lt?Oq??!*???6?@)?????~??1'hc??2@:Preprocessing2s
<Iterator::Model::MaxIntraOpParallelism::FiniteSkip::Prefetch????c??!,ȿ!??@)????c??1,ȿ!??@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismg???d???!?Z??@)nm?y??h?1V~??+??:Preprocessing2F
Iterator::Model^??_>??!??'?er@)??v?ӂg?1?`fB????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.