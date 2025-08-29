#Melanie Weilert
#November 2020
#Stowers Institute, Zeitlinger Lab
#Purpose: Import an ATAC-seq .BAM file and deduplicate, correct Tn5 bias, and modify fragment sizes to provide higher quality coverage.

suppressPackageStartupMessages(library(optparse, warn.conflicts=F, quietly=T))
suppressPackageStartupMessages(library(testit, warn.conflicts=F, quietly=T))

option_list <- list(
  make_option(c("-f", "--file"),
              type="character",
              help="Path of BAM file to process"),
  make_option(c("-r", "--r_file_name_output"),
              type="character",
              default=NA,
              help="Name for resulting R object. This .RDATA file could be deduplicated, fragment filtered, and Tn5 bias corrected based on other associated parameters. If is not NA, then create file. [default = NA]"),
  make_option(c("-w", "--bigwig_file_name_output"),
              type="character",
              default=NA,
              help="Name for resulting .BW object. This .BW file could be deduplicated, fragment filtered, and Tn5 bias corrected based on other associated parameters. If is not NA, then create file. [default = NA]"),
  make_option(c("-s", "--single_ended"),
              action="store_true",
              default=FALSE,
              help="Were these files sequenced in single-ended mode? The default is FALSE"),
  make_option(c("-e", "--fragment_length_extension"),
              type="integer",
              default=NA,
              help="If --single_ended=T, then set a fragment length extension to resize the aligned ATAC-seq reads. Keep in mind that this extension occurs BEFORE cutsite correction [default = NA]."),
  make_option(c("-d", "--deduplicate_file"),
              type="logical",
              default=TRUE,
              help="Deduplicate the file based on *previously marked* duplicates by default Picard MarkDuplicates in pipeline. Default = T"),
  make_option(c("-t", "--tn5_cut_correction"),
              type="logical",
              default=TRUE,
              help="Correct reads based on Tn5 transposase cuts (Buenrostro 2013). Positive strands will be offset by +4bp and negative strands will be offset by -5bp. Note that this is not the same as the Tn5 bias based on sequences. Default = T"),
  make_option(c("-c", "--cut_site_export"),
              type="logical",
              default=FALSE,
              help="When exporting the bigwig files, should the bigwigs be exported as pileups or cut sites? Default = F"),
  make_option(c("-l", "--minimum_fragment_size"),
              type="integer",
              default=10,
              help="Minimum fragment size. Anything smaller will be removed. Anything less than 10 will error out because it cannot withstand the Tn5 cut correction. Default = 10"),
  make_option(c("-u", "--maximum_fragment_size"),
              type="integer",
              default=600,
              help="Maximum fragment size. Anything larger will be removed. Default = 600"),
  make_option(c("-m", "--deduplicate_based_on_mosaic_barcoding"),
              action="store_true",
              default=FALSE,
              help="Instead of deduplicating based on the Picard marks, deduplicate based on the .bam file names. This is a subselection that still requires --deduplicate_file to go through. The default is FALSE")
)

opt <- parse_args(OptionParser(option_list=option_list, usage = "Rscript %prog [options]"))

# opt$file<-'/l/Zeitlinger/ZeitlingerLab/pipeline/data/lab_data/bam/zelda_and_nucleosomes/dme_native_cic6_25to3_atac_1.bam'
#opt$deduplicate_based_on_mosaic_barcoding<-FALSE

#Functions in code
keep_dovetailed_reads<-function(bam, has_unique_reads = FALSE){
  message("Resolving dovetailed discordant reads, keeping discordants based on mapping distance (dovetailing) and removing discordants based on orientation.")

  # First, subset by strand situation
  bam.pos<-bam[strand(bam)=="+"]
  bam.neg<-bam[strand(bam)=="-"]

  #Next, filter the truly discordant reads
  pos.is_discordant_to_keep.idx<-((start(bam.pos@first)-end(bam.pos@last)) < 0) & (seqnames(bam.pos@first)==seqnames(bam.pos@last))
  neg.is_discordant_to_keep.idx<-((start(bam.neg@last)-end(bam.neg@first)) < 0) & (seqnames(bam.neg@first)==seqnames(bam.neg@last))

  #Keep all the correct reads
  bam.pos<-bam.pos[pos.is_discordant_to_keep.idx]
  bam.neg<-bam.neg[neg.is_discordant_to_keep.idx]

  #Take the correct coordinates and save as GRanges.
  corrected.pos.gr<-GRanges(seqnames = seqnames(bam.pos),
                            ranges = IRanges(start = start(bam.pos@first),
                                             end = end(bam.pos@last)),
                            strand = strand(bam.pos))
  corrected.neg.gr<-GRanges(seqnames = seqnames(bam.neg),
                            ranges = IRanges(start = start(bam.neg@last),
                                             end = end(bam.neg@first)),
                            strand = strand(bam.neg))
  if(has_unique_reads){
    corrected.pos.gr$name<-bam.pos@first@elementMetadata$qname
    corrected.neg.gr$name<-bam.neg@first@elementMetadata$qname
  }

  bam_filtered.gr<-c(corrected.pos.gr, corrected.neg.gr)
  message(length(bam_filtered.gr), " reads kept after discordant resolution.")
  return(bam_filtered.gr)
}

#Add assertions to ensure proper naming conventions.
assert("Output R file does not end with .granges.rds. Check inputs.", any(is.na(opt$r_file_name_output), grepl(".granges.rds", opt$r_file_name_output)))
assert("Output .bw file does not end with .granges.rds. Check inputs.", any(is.na(opt$bigwig_file_name_output), grepl(".bw", opt$bigwig_file_name_output)))
assert("Please select a larger minimum fragment size that can withstand the 9bp Tn5 cut correction.", opt$minimum_fragment_size>=10)
assert("Single-ended fragments must have an extension length specified.", ifelse(opt$single_ended, !is.na(opt$fragment_length_extension), T))

# # Example options for developing code
# opt$file<-"/n/projects/mw2098/analysis/rotational_periodicity/bam/GSE66386/atac_saccer3_dedup_merged.bam"
# opt$bigwig_file_name_output<-"/n/projects/mw2098/analysis/rotational_periodicity/bw/GSE66386/atac_saccer3_dedup_merged_cuts.bw"
# opt$deduplicate_file<-FALSE
# opt$cut_site_export<-TRUE

suppressPackageStartupMessages(library(Rsamtools))
suppressPackageStartupMessages(library(GenomicAlignments))
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(rtracklayer))
id <- paste0("[", opt$name, "] ")
stopifnot(file.exists(opt$file))

#Import .BAM file reads
if(opt$single_ended) {
  message("Reading single ended ATAC-seq BAM: ", opt$file)

  #Remove reads marked by Picard and remove any alignments that are secondary (multimapping...should not happen under current bowtie2 settings)
  if(opt$deduplicate_file){
    if(opt$deduplicate_based_on_mosaic_barcoding){
      message("Reading BAM: ", opt$file)
      bam <- readGAlignments(opt$file, param=ScanBamParam(what=scanBamWhat()))
      message(pn(length(bam)), " reads")

      #Convert to GRanges, maintain information from GAlignments
      bam.gr<-granges(bam)
      bam.gr$seq<-bam@elementMetadata$seq
      bam.gr$qual<-bam@elementMetadata$qual
      bam.gr$mrnm<-"*" #Set this to be blank in single-ended reads
      bam.gr$barcode <- mcols(bam)$qname
      names(bam.gr) <- NULL

      message("Removing barcode duplicates...")
      grl   <- split(bam.gr, mcols(bam)$qname)
      keep <- !duplicated(grl)
      bam_filtered.gr <- grl[keep] %>%
        unlist(use.names=FALSE)

    }
    else{
      param <- ScanBamParam(what = "qname", flag=scanBamFlag(isDuplicate=FALSE))
      bam <- readGAlignments(opt$file, param=param)
      message(id, length(bam), " fragments read. If deduplicate==T, then these are duplicates. If not, these are raw reads.")
      bam_filtered.gr<-granges(bam) %>% resize(., opt$fragment_length_extension, 'start')
    }
  } else {
    param <- ScanBamParam(what = "qname")
    bam <- readGAlignments(opt$file, param=param)
    message(id, length(bam), " fragments read. If deduplicate==T, then these are duplicates. If not, these are raw reads.")
    bam_filtered.gr<-granges(bam) %>% resize(., opt$fragment_length_extension, 'start')
  }

} else {
  message(id, "Reading paired-end ATAC-seq BAM: ", opt$file)
  #Remove unpaired reads, reads marked by Picard and remove any alignments that are secondary (multimapping...should not happen under current bowtie2 settings)
  if(opt$deduplicate_file){
    if(opt$deduplicate_based_on_mosaic_barcoding){
      message(id, "Reading paired-end BAM: ", opt$file)
      bam <- readGAlignmentPairs(opt$file, param=ScanBamParam(what="qname"))
      bam.gr<-keep_dovetailed_reads(bam, has_unique_reads = T)
      bam.gr$barcode <- gsub("^(.*)_pair_.*$", "\\1", bam.gr$name)

      message("Removing barcode duplicates...")
      grl   <- split(bam.gr, bam.gr$barcode)
      keep <- !duplicated(grl)
      bam_filtered.gr <- grl[keep] %>%
        unlist(use.names=FALSE)
    }
    else{
      param <- ScanBamParam(what = "qname", flag=scanBamFlag(isDuplicate=FALSE))
      bam <- readGAlignmentPairs(opt$file, param=param)
      message(id, length(bam), " fragments read. If deduplicate==T, then these are duplicates. If not, these are raw reads.")
      bam_filtered.gr<-keep_dovetailed_reads(bam)
    }
  } else {
    param <- ScanBamParam(what = "qname")
    bam <- readGAlignmentPairs(opt$file, param=param)
    message(id, length(bam), " fragments read. If deduplicate==T, then these are duplicates. If not, these are raw reads.")
    bam_filtered.gr<-keep_dovetailed_reads(bam)
  }

}

# Filter by fragment size
message("Filtering by fragment size. Max = ", opt$maximum_fragment_size, ", Min = ", opt$minimum_fragment_size)
widths<-width(bam_filtered.gr)
keep<-(widths >= opt$minimum_fragment_size) & (widths <= opt$maximum_fragment_size)
bam_filtered.gr<-bam_filtered.gr[keep]
message(id, length(bam_filtered.gr), " fragments within size restrictions.")

if(opt$tn5_cut_correction){
  # Accommodate Tn5 cuts according to (Buenrostro 2013)
  # For reads on the positive strand, shift them by 4bp downstream.
  # For reads aligned on the negative strand, shift them by 5 bp upstream (downstream relative to the 5' sequence direction of the negative strand)
  # Look at figure 1 here for more details about Tn5 nicking: https://currentprotocols.onlinelibrary.wiley.com/doi/10.1002/0471142727.mb2129s109
  
  #Remember, because the nicks happen on both ends of a paired end fragment, then this means we have to correct each 5' end per fragment.
  
  #Shift positive "start" sites +4 bp
  start(bam_filtered.gr)<-start(bam_filtered.gr) + 4
  
  #Shift negative "start" sites -4 bp
  #This is in contrast to the ATAC-seq ENCODE pipeline approach, and is aligned more with the cutsite footprinting approach used by TOBIAS.
  #After discussions with the Kundaje Lab, this approach is more in-line with representation of cut-sites as a true -4/+4 feature with the base in between representing a cut site.
  end(bam_filtered.gr)<-end(bam_filtered.gr) - 4
}

message(id, "Saving ", length(bam_filtered.gr), " fragments...")

#Save reads as a .granges.rds file
if(!is.na(opt$r_file_name_output)){
  message("Saving .granges.rds file as: ", opt$r_file_name_output)
  temp_r_name<-gsub(pattern = "^.granges.rds$", replacement = "_temp.granges.rds", opt$r_file_name_output)
  saveRDS(bam_filtered.gr, file=temp_r_name)
  file.rename(from = temp_r_name, to = opt$r_file_name_output) #rename once writing is done to prevent snakemake errors mid-saving the file
}
if(!is.na(opt$bigwig_file_name_output)){

  #Decide whether to export as sequence-uncorrected cut sites or pileup.
  if(opt$cut_site_export){
    if(opt$single_ended){
      bam_final.cov<-coverage(resize(bam_filtered.gr, 1, 'start'))
    }else{
      bam_final.cov<-coverage(c(resize(bam_filtered.gr, 1, 'start'),
                                resize(bam_filtered.gr, 1, 'end')))
    }
  }else{
    bam_final.cov<-coverage(bam_filtered.gr)
  }

  message("Saving .bw file as: ", opt$bigwig_file_name_output)
  # temp_bw_name<-gsub(pattern = "^.bw$", replacement = "_temp.bw", opt$bigwig_file_name_output)
  rtracklayer::export(object = bam_final.cov, con = opt$bigwig_file_name_output, format = "BigWig")
  # file.rename(from = temp_bw_name, to = opt$bigwig_file_name_output) #rename once writing is done to prevent snakemake errors mid-saving the file
}

message("Done! =)")

#Note
#########################################################################################################
#When importing the BAM file from GAlignments, account for automatically assigned NA values.
#This will prevent `[W::sam_parse1] urecognized mate reference name; treated as unmapped` error message upon export.
#Kudos to: https://github.com/paul-shannon/igvR/issues/4
## mrnm (mate reference name)
# meta <- bam_uniq.gr@elementMetadata
# meta$mrnm <- "*"
# bam_uniq.gr@elementMetadata <- meta
#########################################################################################################
