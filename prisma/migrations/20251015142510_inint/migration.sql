/*
  Warnings:

  - You are about to drop the column `db` on the `FailDocs` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "FailDocs" DROP COLUMN "db",
ADD COLUMN     "dob" TEXT;
