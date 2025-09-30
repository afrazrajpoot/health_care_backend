/*
  Warnings:

  - Added the required column `gcsFileLink` to the `Document` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE "Document" ADD COLUMN     "gcsFileLink" TEXT NOT NULL;
